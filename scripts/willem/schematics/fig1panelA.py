import numpy as np

from neat import MorphTree, PhysTree, NeuronSimTree
from datarep import paths
from datarep.matplotlibsettings import *

from channels import channels_hay


TAU_R_AMPA = .2     # ms
TAU_D_AMPA = 3.     # ms
NMDA_RATIO = 2.
TAU_R_NMDA = 1.     # ms
TAU_D_NMDA = 43.    # ms
E_REV      = 0.     # mV
W_CONTEXT  = 0.0005 # uS
W_FEEDF    = 0.006  # uS

newcolours = colours[2:]
morphcolours = ['DarkGrey', colours[0], colours[1]]


def plotMorphology(ax=None, locs=None):
    if ax is None:
        pl.figure(figsize=(4,6))
        ax = pl.gca()
        pshow = True
    else:
        pshow = False

    cmap = mcolors.ListedColormap(morphcolours)

    mtree = MorphTree(paths.morph_path + 'cell1_simplified.swc')

    nodes1 = [n for n in mtree.getNodesInBasalSubtree() if mtree.pathLength((n.index,.5), (1,.5)) > 100.]
    nodes2 = [n for n in mtree.getNodesInApicalSubtree() if mtree.pathLength((n.index,.5), (1,.5)) > 200.]
    nodes3 = [n for n in mtree.getNodesInBasalSubtree() if mtree.pathLength((n.index,0.), (1,.5)) < 100.]
    nodes4 = [n for n in mtree.getNodesInApicalSubtree() if mtree.pathLength((n.index,0.), (1,.5)) < 200.]
    nodes0 = list(set(mtree.nodes) - set(nodes1 + nodes2 + nodes3))

    cs =      {n.index: 0.125 for n in nodes0}
    cs.update({n.index: 0.500 for n in nodes1})
    cs.update({n.index: 0.500 for n in nodes2})
    cs.update({n.index: 0.875 for n in nodes3})
    cs.update({n.index: 0.875 for n in nodes4})

    # markers = [{'marker': 's', 'c': newcolours[ii], 'mec': 'k', 'ms': markersize} for ii in range(len(locs))]
    markers = [{'marker': 'v', 'c': colours[0], 'mec': 'k', 'ms': markersize} for ii in range(len(locs))]
    # marklabels = {ii: str(ii) for ii in range(len(locs))}
    marklabels = []
    plotargs = {'lw': lwidth/1.3}
    mtree.plot2DMorphology(ax, use_radius=0, plotargs=plotargs, sb_draw=0,
                                cs=cs, cmap=cmap,
                                marklocs=locs, locargs=markers, marklabels=marklabels, lims_margin=0.01)

    if pshow:
        pl.show()


def burstTrain(rate, tmax, tstart=4., nspk=10, tspread=10., seed1=None, seed2=None):
    t_bursts = poissonTrain(rate, tmax, tstart=tstart, seed=seed1)
    if seed2 is not None:
        np.random.seed(seed2)
    t_spks = []
    for tb in t_bursts:
        n_spk = np.random.poisson(lam=nspk)
        t_spk = tb + tspread * np.random.randn(n_spk)
        t_spks.extend(t_spk.tolist())
    return np.sort(t_spks)


def getL5Pyramid():
    """
    Return a minimal model of the L5 pyramid for BAC-firing
    """
    # load the morphology
    phys_tree = PhysTree(paths.morph_path + 'cell1_simplified.swc')

    # set specific membrane capacitance and axial resistance
    phys_tree.setPhysiology(1., # Cm [uF/cm^2]
                            100./1e6, # Ra[MOhm*cm]
                            node_arg=[phys_tree[1]])
    # set specific membrane capacitance and axial resistance
    phys_tree.setPhysiology(2., # Cm [uF/cm^2]
                            100./1e6, # Ra[MOhm*cm]
                            node_arg=[n for n in phys_tree if not phys_tree.isRoot(n)])

    # channels present in tree
    Kv3_1  = channels_hay.Kv3_1()
    Na_Ta  = channels_hay.Na_Ta()
    Ca_LVA = channels_hay.Ca_LVA()
    Ca_HVA = channels_hay.Ca_HVA()
    h_HAY  = channels_hay.h_HAY()

    # soma ion channels [uS/cm^2]
    phys_tree.addCurrent(Kv3_1,  0.766    *1e6, -85., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Na_Ta,  1.71     *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Ca_LVA, 0.00432  *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(Ca_HVA, 0.000567 *1e6,  50., node_arg=[phys_tree[1]])
    phys_tree.addCurrent(h_HAY,  0.0002   *1e6, -45., node_arg=[phys_tree[1]])
    phys_tree.setLeakCurrent(0.0000344 *1e6, -90., node_arg=[phys_tree[1]])

    # basal ion channels [uS/cm^2]
    phys_tree.addCurrent(h_HAY, 0.0002 *1e6, -45., node_arg='basal')
    phys_tree.setLeakCurrent(0.0000535 *1e6, -90., node_arg='basal')

    # apical ion channels [uS/cm^2]
    phys_tree.addCurrent(Kv3_1, 0.000298 *1e6, -85., node_arg='apical')
    phys_tree.addCurrent(Na_Ta, 0.0211   *1e6,  50., node_arg='apical')
    phys_tree.addCurrent(Ca_LVA, lambda x: 0.0198*1e6   if (x>685. and x<885.) else 0.0198*1e-2*1e6,   50., node_arg='apical')
    phys_tree.addCurrent(Ca_HVA, lambda x: 0.000437*1e6 if (x>685. and x<885.) else 0.000437*1e-1*1e6, 50., node_arg='apical')
    phys_tree.addCurrent(h_HAY,  lambda x: 0.0002*1e6 * (-0.8696 + 2.0870 * np.exp(x/323.)),          -45., node_arg='apical')
    phys_tree.setLeakCurrent(0.0000447*1e6, -90., node_arg='apical')

    return phys_tree


def spikeBurst(tb, tspread, n_spk):
    t_spk = tb + tspread * np.random.randn(n_spk)
    t_spk.sort()
    return t_spk


def simNMDA(axes=None):

    # locs = [(1,.5)  , # soma
    #         (167,.8), # 400 um
    #         (192,.8), # 430 um
    #         (349,.8), # 310 um
    #         # (11,.8) , # 200 um
    #         (136,.8), # 220 um
    #         # (62,.8) , # 200 um
    #         # (103,.8), # 200 um
    #         (56,.8) , # 260 um
    #         (139,.8), # 270 um
    #         (327,.8), # 330 um
    #         ]

    locs = [(1,.5)  , # soma
            (167,.8), # 400 um
            (136,.8), # 220 um
            (139,.8), # 270 um
            (192,.8), # 430 um
            (327,.8), # 330 um
            (349,.8), # 310 um
            (56,.8) , # 260 um
            ]

    if axes is None:
        pl.figure('morph', figsize=(4,3))
        ax_morph = pl.gca()

        pl.figure('dend', figsize=(3,12))
        gs = GridSpec(len(locs)-1, 1)
        axes_context = [pl.subplot(gs[ii,0]) for ii in range(len(locs)-1)]

        pl.figure('soma', figsize=(3,6))
        ax_t1 = noFrameAx(pl.subplot(411))
        ax_feedf = noFrameAx(pl.subplot(412))
        ax_t2 = noFrameAx(pl.subplot(413))
        ax_vm = myAx(pl.subplot(414))

        pshow = True

    else:
        axes_context = axes['context']
        ax_morph = axes['morph']

        ax_t1, ax_feedf, ax_t2, ax_vm = axes['volt']
        ax_t1 = noFrameAx(ax_t1)
        ax_t2 = noFrameAx(ax_t2)

        pshow = False

    plotMorphology(ax_morph, locs[1:])

    phys_tree = getL5Pyramid()
    sim_tree = phys_tree.__copy__(new_tree=NeuronSimTree())

    vm_store = []
    for ii in range(2, len(locs)+1):
        sim_tree.initModel(t_calibrate=200.)

        sim_tree.addIClamp(locs[0], -0.1, -200., 500.)
        sim_tree.addDoubleExpSynapse(locs[0], tau1=TAU_R_AMPA, tau2=TAU_D_AMPA, e_r=E_REV)
        sim_tree.setSpikeTrain(0, W_FEEDF, [78.,84.])

        for kk, jj in enumerate(range(1,ii)):
            sim_tree.addDoubleExpNMDASynapse(locs[jj], tau1=TAU_R_AMPA, tau2=TAU_D_AMPA, tau1_nmda=TAU_R_NMDA, tau2_nmda=TAU_D_NMDA, e_r=E_REV, nmda_ratio=NMDA_RATIO)
            sim_tree.setSpikeTrain(jj, W_CONTEXT, spikeBurst(50., 5., 40))

        sim_tree.storeLocs(locs, name='rec locs')
        res = sim_tree.run(200.)

        sim_tree.deleteModel()

        vm_store.append(res['v_m'][0])

        # pl.figure(figsize=(16,3))
        # gs = GridSpec(1, len(locs))

        # for kk, loc in enumerate(locs):
        #     ax = pl.subplot(gs[0,kk])
        #     ax.plot(res['t'], res['v_m'][kk])
        #     ax.set_ylim((-90.,0.))

        # pl.tight_layout()
        # pl.show()

    ax = noFrameAx(axes_context[0])
    ax.text(.5,.5, r'Contextual input', c=colours[0], ha='center', va='center', fontsize=labelsize)
    ax.set_xlim((0.,1.))
    ax.set_ylim((0.,1.))

    for ii, vm in enumerate(res['v_m'][1:]):
        ax = axes_context[ii+1]
        ax.plot(res['t'], vm, lw=lwidth, c=colours[0])

        ax.set_ylim((-90.,0.))
        ax.set_xlim((0.,200.))
        ax.set_xticks([])
        ax.set_yticks([])

    ax_t1.text(.5,.5, r'Feedforward input', c=colours[1], ha='center', va='center', fontsize=labelsize)
    ax_t1.set_xlim((0.,1.))
    ax_t1.set_ylim((0.,1.))

    plotSpikeRaster(ax_feedf, [[78.,84.]], cs=colours[1], plotargs={'lw': lwidth})
    ax_feedf.set_xlim((0.,200.))

    ax_t2.text(.5,.5, r'Context-dependent response', c='k', ha='center', va='center', fontsize=labelsize)
    ax_t2.set_xlim((0.,1.))
    ax_t2.set_ylim((0.,1.))

    for ii, vm in enumerate(vm_store):
        cc = 'DarkGrey' if ii < len(vm_store)-1 else 'k'
        ax_vm.plot(res['t'], vm, lw=lwidth, c=cc)

    ax_vm.set_ylim((-90.,0.))
    ax_vm.set_xlim((0.,200.))

    # ax_vm.axvline(78., ls='--', lw=lwidth*.7, c=colours[1])
    # ax_vm.axvline(84., ls='--', lw=lwidth*.7, c=colours[1])

    drawScaleBars(ax_vm, xlabel=r'ms', ylabel=r'mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

    if pshow:
        pl.show()


def plotFigure():


    xcoords = getXCoords([0.2, 0.2, 0.05, 0.4, 0.05, 0.2, 0.2, 0.5, 0.15])

    fig = pl.figure('Bias schema', figsize=(7,3))

    gs0 = pl.GridSpec(4,1)
    gs0.update(top=0.9, bottom=0.1, left=xcoords[0], right=xcoords[1], hspace=0.2, wspace=0.01)

    gsm = pl.GridSpec(1,1)
    gsm.update(top=0.9, bottom=0.2, left=xcoords[2], right=xcoords[3], hspace=0.2, wspace=0.01)

    gs1 = pl.GridSpec(4,1)
    gs1.update(top=0.9, bottom=0.1, left=xcoords[4], right=xcoords[5], hspace=0.2, wspace=0.01)

    gsv = pl.GridSpec(8,1)
    gsv.update(top=0.95, bottom=0.1, left=xcoords[6], right=xcoords[7], hspace=0.2, wspace=0.01)

    axesc = [pl.subplot(gs0[ii,0]) for ii in range(4)] +  [pl.subplot(gs1[ii,0]) for ii in range(4)]

    axm = pl.subplot(gsm[0,0])

    axt1 = pl.subplot(gsv[0,0])
    axin = pl.subplot(gsv[1,0])
    axt2 = pl.subplot(gsv[2,0])
    axvm = pl.subplot(gsv[3:,0])

    axes = {}
    axes['context'] = axesc
    axes['morph'] = axm
    axes['volt'] = [axt1, axin, axt2, axvm]

    simNMDA(axes=axes)

    pl.savefig(paths.fig_path + "biasadapatation_fig1A.svg", transparent=True)

    pl.show()


if __name__ == "__main__":
    # simNMDA()
    plotFigure()





