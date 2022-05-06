
def plot_receptive_fields(n_h1, algo='sm', reduced=True):
    storage = load_optimization_data(n_h1, algo, reduced=reduced)

    sampler = samplers.NTaskSampler('EMNIST',
                        n_per_batch=25, n_per_epoch=1)
    n_inp = sampler.get_input_dim()

    for ll, resdict in enumerate(storage):
        ws = resdict['w_final']


        sampler.set_tasks(resdict['task'])
        xdata, xtask, xtarget = next(iter(sampler))

        pl.figure('Task %d = '%ll + str(resdict['task']) + ' | perf = %.2f'%resdict['perf'][-1], figsize=(16,8))
        gs_ = GridSpec(5,5)
        gs_.update(top=0.95, bottom=0.05, left=0.05, right=0.46, hspace=0.05, wspace=0.05)
        gs = GridSpec(5,5)
        gs.update(top=0.95, bottom=0.05, left=0.54, right=0.95, hspace=0.05, wspace=0.05)

        w_fields  = np.dot(ws[1], ws[0])

        for kk, w_vec in enumerate(w_fields):
            ii, jj = kk//5, kk%5

            ax = pl.subplot(gs[ii,jj])
            ax.imshow(utils.to_image_mnist(w_vec))
            ax.set_xticks([])
            ax.set_yticks([])


        for kk, (x_vec, x_target) in enumerate(zip(xdata, xtarget)):
            ii, jj = kk//5, kk%5

            ax = pl.subplot(gs_[ii,jj])
            ax.set_title('target = %d'%x_target)
            ax.imshow(utils.to_image_mnist(x_vec))
            ax.set_xticks([])
            ax.set_yticks([])



        pl.show()