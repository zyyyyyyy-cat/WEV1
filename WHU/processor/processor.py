import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    # `centers[m]` is the per-identity center of modality m, shape [P, D].
    # GPD-SC needs these modality-specific centers for its sub-center update.
    return loss, ids, g_cent, centers


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


class SubCenterPrototypeBank(nn.Module):
    """GPD-SC prototype memory: K sub-centers per identity.

    The K sub-centers are intended to carry the RGB/NIR/TIR modality centers
    of the same identity.  Updates use the modality center of the current
    samples (not the cross-modal identity center `g_cent`), so the sub-centers
    keep their modality semantics instead of collapsing to one point.
    """

    def __init__(self, num_classes, feat_dim, num_subcenters=3, momentum=0.2,
                 device='cuda', count_ema=False, ema_min=0.1, ema_max=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.K = int(num_subcenters)
        self.m = momentum
        self.count_ema = count_ema
        self.ema_min = ema_min
        self.ema_max = ema_max
        self.register_buffer('sub_centers',
                             torch.zeros(num_classes, self.K, feat_dim, device=device))
        self.register_buffer('counts', torch.zeros(num_classes, self.K, device=device))

    @torch.no_grad()
    def initialize(self, centers):
        """centers: [C, K, D], already L2-normalized."""
        if centers.shape != self.sub_centers.shape:
            raise ValueError('sub-center shape mismatch: {} vs {}'.format(
                tuple(centers.shape), tuple(self.sub_centers.shape)))
        self.sub_centers.copy_(centers.float())
        self.counts.fill_(1.0)

    @torch.no_grad()
    def update(self, ids_local, mod_ids_local, centers_local, ddp=True):
        """Update one sub-center per (identity, modality) pair.

        ids_local:    [G] identity ids
        mod_ids_local:[G] modality index in [0, K)
        centers_local:[G, D] modality center for each pair
        """
        ids_local = ids_local.to(self.sub_centers.device).contiguous()
        mod_ids_local = mod_ids_local.to(self.sub_centers.device).contiguous()
        centers_local = centers_local.to(self.sub_centers.device).float().contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            mod_list = [torch.empty_like(mod_ids_local) for _ in range(world)]
            center_list = [torch.empty_like(centers_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(mod_list, mod_ids_local)
            dist.all_gather(center_list, centers_local)
            ids_all = torch.cat(ids_list, dim=0).long()
            mod_all = torch.cat(mod_list, dim=0).long()
            centers_all = torch.cat(center_list, dim=0).float()
        else:
            ids_all = ids_local.long()
            mod_all = mod_ids_local.long()
            centers_all = centers_local.float()

        C, K, D = self.sub_centers.shape
        flat = self.sub_centers.view(C * K, D)
        count_flat = self.counts.view(C * K)
        idx = ids_all * K + mod_all      # [G_all]

        sums = torch.zeros_like(flat)
        counts = torch.zeros(C * K, device=flat.device)
        sums.index_add_(0, idx, centers_all)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))

        mask = counts > 0
        new_centers = F.normalize(sums[mask] / counts[mask].unsqueeze(1), dim=1)
        old = flat[mask]
        bank_count = count_flat[mask]
        if self.count_ema:
            m_eff = torch.clamp(1.0 / (bank_count + 1.0),
                                min=self.ema_min, max=self.ema_max)
            # An uninitialized sub-center should be set directly, not blended
            # with a zero vector.
            m_eff = torch.where(bank_count <= 0,
                                torch.ones_like(m_eff), m_eff)
        else:
            m_eff = torch.full_like(bank_count, float(self.m))
        upd = F.normalize((1.0 - m_eff).unsqueeze(1) * old
                          + m_eff.unsqueeze(1) * new_centers, dim=1)
        flat[mask] = upd
        count_flat[mask] += 1.0


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


@torch.no_grad()
def warm_start_subcenters(model, bank, loaders, device, logger=None):
    """Initialize K sub-centers per identity with per-modality class means.

    `loaders` is the tri-modal deterministic training loader list, ordered as
    [RGB, NIR, TIR].  Missing (identity, modality) groups fall back to the
    identity mean over valid modalities.
    """
    if not loaders:
        return 0
    if len(loaders) != bank.K:
        raise ValueError('warm-start loaders ({}) != sub-centers ({})'.format(
            len(loaders), bank.K))
    model.eval()
    C, K, D = bank.num_classes, bank.K, bank.feat_dim
    centers, valids = [], []
    for mode, loader in enumerate(loaders, start=1):
        sums = torch.zeros(C, D, device=device)
        cnts = torch.zeros(C, device=device)
        for batch in loader:
            img = batch[0].to(device)
            pid = torch.as_tensor(batch[1], dtype=torch.long, device=device)
            feat = model(img, mode=mode)
            feat = F.normalize(feat.float(), dim=1)
            sums.index_add_(0, pid, feat)
            cnts.index_add_(0, pid, torch.ones_like(pid, dtype=cnts.dtype))
        valid = cnts > 0
        center = torch.zeros_like(sums)
        center[valid] = F.normalize(sums[valid] / cnts[valid].unsqueeze(1), dim=1)
        centers.append(center)
        valids.append(valid)

    id_sum = torch.zeros(C, D, device=device)
    id_cnt = torch.zeros(C, device=device)
    for center, valid in zip(centers, valids):
        id_sum += center * valid.unsqueeze(1).float()
        id_cnt += valid.float()
    id_mean = F.normalize(
        id_sum / torch.clamp(id_cnt, min=1.0).unsqueeze(1), dim=1)
    for m in range(K):
        missing = ~valids[m]
        if missing.any():
            centers[m][missing] = id_mean[missing]

    bank.initialize(torch.stack(centers, dim=1))
    model.train()
    if logger is not None:
        logger.info('GPD-SC warm start: {}/{} identities, K={}'.format(
            int((id_cnt > 0).sum().item()), C, K))
    return int((id_cnt > 0).sum().item())


def GPD_SC(feat, target, bank: SubCenterPrototypeBank, ids, g_cent,
           num_modalities, tau=0.03):
    """GPD with K sub-centers per identity (max over sub-centers)."""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat.float(), dim=1)
    with torch.no_grad():
        P = bank.sub_centers.float().clone()
        norms = P.norm(dim=2)                    # [C, K]
        need = norms[ids].sum(dim=1) < 1e-6      # [P]
        if need.any():
            P[ids[need]] = g_cent[need].float().unsqueeze(1).expand(
                -1, bank.K, -1)
        P = F.normalize(P, dim=2)

    # [N, C, K] -> max over K -> [N, C]
    logits = torch.einsum('nd,ckd->nck', f, P) / tau
    logits = logits.max(dim=2).values
    return F.cross_entropy(logits, target_rep)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_normal_loaders,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(local_rank)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    gpd_type = getattr(cfg.MODEL, 'GPD_TYPE', 'single').lower().strip()
    if gpd_type not in ('single', 'subcenter'):
        raise ValueError("MODEL.GPD_TYPE must be 'single' or 'subcenter', got {}".format(gpd_type))
    if is_main:
        logger.info('Feature dimension: {}; identities: {}; GPD type: {}'.format(
            feat_dim, num_classes, gpd_type))
    if use_aux_modules:
        if gpd_type == 'subcenter':
            num_modalities_cfg = len(cfg.DATASETS.MODALITIES)
            if cfg.MODEL.GPD_SUB_CENTERS != num_modalities_cfg:
                raise ValueError(
                    "GPD-SC uses modality-indexed sub-centers: "
                    "GPD_SUB_CENTERS ({}) must equal the number of modalities "
                    "({}). Use GPD_TYPE='single' for the K=1 baseline.".format(
                        cfg.MODEL.GPD_SUB_CENTERS, num_modalities_cfg))
            proto_bank = SubCenterPrototypeBank(
                num_classes=num_classes,
                feat_dim=feat_dim,
                num_subcenters=cfg.MODEL.GPD_SUB_CENTERS,
                momentum=cfg.MODEL.GPD_MOMENTUM,
                device=device,
                count_ema=cfg.MODEL.GPD_COUNT_EMA,
                ema_min=cfg.MODEL.GPD_EMA_MIN,
                ema_max=cfg.MODEL.GPD_EMA_MAX,
            )
        else:
            proto_bank = PrototypeBank(
                num_classes=num_classes,
                feat_dim=feat_dim,
                momentum=cfg.MODEL.GPD_MOMENTUM,
                device=device,
            )
        if cfg.MODEL.GPD_WARMSTART and gpd_type == 'subcenter':
            if cfg.TEST.NECK_FEAT != 'before':
                raise ValueError(
                    "GPD-SC warm start expects TEST.NECK_FEAT='before' so that "
                    "the sub-centers use the same pre-BN CLS as GPD.")
            # DataLoader iterator creation draws the worker base seed from the
            # global torch RNG.  Save/restore it so that warm-start on/off runs
            # keep the same training augmentation stream.
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            if dist_on:
                if is_main:
                    warm_start_subcenters(model_meta, proto_bank,
                                          train_normal_loaders, device, logger)
                dist.barrier()
                dist.broadcast(proto_bank.sub_centers, src=0)
                dist.broadcast(proto_bank.counts, src=0)
            else:
                warm_start_subcenters(model_meta, proto_bank,
                                      train_normal_loaders, device, logger)
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    group_size = cfg.DATALOADER.NUM_INSTANCE

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        n_iter = 0
        for n_iter, (imgs, vid, camids) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            with amp.autocast(enabled=True):
                cls_score, global_feat, feat = model(imgs, target, camids)

                loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])

                if use_aux_modules:
                    loss_pca, ids, g_cent, centers = PCA(
                        global_feat,
                        target,
                        num_modalities=num_modalities,
                        group_size=group_size,
                    )
                    meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                    if gpd_type == 'subcenter':
                        loss_gpd = GPD_SC(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=cfg.MODEL.GPD_TAU,
                        )
                    else:
                        loss_gpd = GPD(
                            global_feat,
                            target,
                            proto_bank,
                            ids,
                            g_cent,
                            num_modalities=num_modalities,
                            tau=cfg.MODEL.GPD_TAU,
                        )
                    meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                    if 'pca' in loss_type:
                        loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                    if 'gpd' in loss_type:
                        loss += loss_gpd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if use_aux_modules:
                with torch.no_grad():
                    if gpd_type == 'subcenter':
                        num_ids = ids.shape[0]
                        centers_all = torch.cat(centers, dim=0)  # [M*P, D]
                        mod_ids = torch.arange(
                            num_modalities, device=device).repeat_interleave(num_ids)
                        proto_bank.update(
                            ids.repeat(num_modalities),
                            mod_ids,
                            centers_all,
                            ddp=cfg.MODEL.DIST_TRAIN,
                        )
                    else:
                        proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)

            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if is_main and use_aux_modules and gpd_type == 'subcenter':
            with torch.no_grad():
                c = proto_bank.counts
                # counts start at 1 after warm start, so >1 means "updated at
                # least once since initialization".
                updated = (c > 1).sum(dim=0)
                mean_count = c.mean(dim=0)
                logger.info(
                    'GPD-SC bank: updated per modality {}, mean count per '
                    'modality {}, count min/mean/max {:.1f}/{:.1f}/{:.1f}'.format(
                        updated.tolist(),
                        [round(float(x), 1) for x in mean_count],
                        c.min().item(), c.mean().item(), c.max().item()))

        if epoch % checkpoint_period == 0:
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )

        if epoch % eval_period == 0:
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode)
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
