# License: BSD 3 clause

import copy
import itertools

from matplotlib import animation, ticker
import matplotlib.pyplot as plt
import numpy as np

standard_colors = ['steelblue', 'darkorange', 'limegreen', 'firebrick',
                   'mediumorchid']


def plot_solver_path(fit_log, matrix_n_param, mu_names, ker_param_names,
                     true_mu=None,
                     true_ker_param=None, min_mu=None,
                     min_ker_param=None, plot_derivatives=False,
                     derivatives_zero=False,
                     axs=None, save=False,
                     filename='image.png', show=False, **kwargs):
    """
    Generic function to plot solver paths for each estimated MHP parameter.

    Parameters
    ----------
    param_updates : `list` of `float`, shape=(d,d, )
        Names of kernel parameters to label the y axis.

    mu_names : `list` of `str`, shape=(d, ), default=`None`
        Names of background rates to label the y axis.

    ker_param_names : `list` of `str`, shape=(d,d, ), default=`None`
        Names of kernel parameters to label the y axis.

    true_mu : `list` of `float`, shape=(d, ), default=`None`
        True vector of background rates.

    true_ker_param : `list` of `float`, shape=(d,d, ), default=`None`
        True tensor of kernel parameters.
        This only makes sense if the fitted MHP model belongs to the same
        parametric class as the ground truth MHP.

    min_mu : `list` of `float`, shape=(d, ), default=`None`
        The vector of background rates which minimizes the LSE.

    min_ker_param : `list` of `float`, shape=(d,d, ), default=`None`
        The tensor of kernel parameters which minimizes the LSE.
        This only makes sense if the fitted MHP model belongs to the same
        parametric class as the ground truth MHP.

    plot_derivatives : `bool`, default=`False`
        if `True`, plot the updates of estimates of the LSE derivative with
        respect to parameters.

    grad_updates : `list` of `float`, shape=(d,d, ), default=`None`
        Names of kernel parameters to label the y axis.

    derivatives_zero : `bool`, default=`False`
        if `True`, display the y=0 line for derivative plots.

    show : `bool`, default=`False`
        if `True`, show the plot.

    Notes
    -----
    true_mu and true_ker_param make sense in the context of data
    simulated from a known ground truth MHP.
    min_mu and min_ker_param are useful in cases where the minimizer
    of the LSE can be known exactly. For example, in the case of a
    1-dimensional MHP with exponential kernel, the minimizer of the LSE is
    known in closed form.
    """
    d = len(matrix_n_param)
    n_iter = [len(fit_log.mu[k])-1 for k in range(d)]

    n_param = d+np.sum(matrix_n_param)

    if plot_derivatives:
        fig_count = 0
        if axs is None:
            kwargs_2 = copy.deepcopy(kwargs)
            if 'pad' in kwargs_2.keys():
                kwargs_2.pop('pad', None)
            fig, axs = plt.subplots(n_param, 2, sharex=True, sharey=False,
                                    **kwargs_2)
        # Mu
        for i in range(d):
            # Parameter
            axs[i, 0].plot(fit_log.mu[i], color=standard_colors[0])
            if true_mu is not None:
                axs[i, 0].hlines(true_mu[i], 0, n_iter[i]+1,
                                 colors=standard_colors[1], linestyles='solid')
            if min_mu is not None:
                axs[i, 0].hlines(min_mu[i], 0, n_iter[i]+1,
                                 colors=standard_colors[2], linestyles='solid')
            axs[fig_count, 0].set(ylabel=mu_names[i]+' updates')

            # Derivative
            axs[i, 1].plot(fit_log.grad_mu[i], color=standard_colors[0])
            if derivatives_zero:
                axs[i, 1].hlines(0., 0, n_iter[i], colors='grey',
                                 linestyles='dashed')
            axs[i, 1].set(ylabel=mu_names[i]+' derivative')

            fig_count += 1

        # Kernel Parameters
        for i, j in itertools.product(range(d), range(d)):
            for ix_param in range(matrix_n_param[i][j]):
                # Parameter
                axs[fig_count, 0].plot([fit_log.ker[i][j][n][ix_param] for n in range(n_iter[i]+1)], color=standard_colors[0])
                if true_ker_param is not None:
                    axs[fig_count, 0].hlines(true_ker_param[i][j][ix_param],
                                             0, n_iter[i]+1,
                                             colors=standard_colors[1],
                                             linestyles='solid')
                if min_ker_param is not None:
                    axs[fig_count, 0].hlines(min_ker_param[i][j][ix_param], 0,
                                             n_iter[i]+1,
                                             colors=standard_colors[2],
                                             linestyles='solid')
                axs[fig_count, 0].set(ylabel=ker_param_names[i][j][ix_param]
                                      + ' updates')

                # Derivative
                axs[fig_count, 1].plot([fit_log.grad_ker[i][j][n][ix_param] for n in range(n_iter[i])], color=standard_colors[0])
                if derivatives_zero:
                    axs[fig_count, 1].hlines(0., 0, n_iter[i],
                                             colors='grey',
                                             linestyles='dashed')
                axs[fig_count, 1].set(ylabel=ker_param_names[i][j][ix_param]
                                      + ' derivative')

                fig_count += 1

        axs[n_param-1, 0].set(xlabel='Iteration')
        axs[n_param-1, 1].set(xlabel='Iteration')

        if "pad" in kwargs:
            pad = kwargs.get('pad', 300)
            fig.tight_layout(pad=pad)

    else:
        n_rows = (n_param // 2) + (n_param % 2)
        if n_rows > 1:
            fig, axs = plt.subplots(n_rows, 2, sharex=True,
                                    sharey=False, **kwargs)
            fig_count = 0
            #   Mu
            for i in range(d):
                row_index, col_index = fig_count // 2, fig_count % 2
                # Parameter
                axs[row_index, col_index].plot(fit_log.mu[i], color=standard_colors[0])
                if true_mu is not None:
                    axs[row_index, col_index].hlines(true_mu[i], 0, n_iter[i]+1, colors=standard_colors[1], linestyles='solid')
                if min_mu is not None:
                    axs[row_index, col_index].hlines(min_mu[i], 0, n_iter[i]+1, colors=standard_colors[2], linestyles='solid')
                axs[row_index, col_index].set(ylabel=mu_names[i] + ' updates')
                fig_count += 1

            #   Kernel Parameters
            for i, j in itertools.product(range(d), range(d)):
                for ix_param in range(len(ker_param_names[i][j])):
                    row_index, col_index = fig_count // 2, fig_count % 2
                    #   Parameter
                    axs[row_index, col_index].plot([fit_log.ker[i][j][n][ix_param] for n in range(n_iter[i]+1)], color=standard_colors[0])
                    if true_ker_param is not None:
                        axs[row_index, col_index].hlines(true_ker_param[i][j][ix_param], 0, n_iter[i]+1, colors=standard_colors[1], linestyles='solid')
                    if min_ker_param is not None:
                        axs[row_index, col_index].hlines(min_ker_param[i][j][ix_param], 0, n_iter[i]+1, colors=standard_colors[2], linestyles='solid')
                    axs[row_index, col_index].set(ylabel=ker_param_names[i][j][ix_param]+' updates')

                    fig_count += 1

            axs[n_rows-1, 0].set(xlabel='Iteration')
            axs[n_rows-1, 1].set(xlabel='Iteration')

            if "pad" in kwargs:
                pad = kwargs.get('pad', 300)
                fig.tight_layout(pad=pad)

        else:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, **kwargs)
            fig_count = 0
            #   Mu
            for i in range(d):
                axs[fig_count % 2].plot(fit_log.mu[i], color=standard_colors[0])
                if true_mu is not None:
                    axs[fig_count % 2].hlines(true_mu[i], 0, n_iter[i]+1, colors=standard_colors[1], linestyles='solid')
                if min_mu is not None:
                    axs[fig_count % 2].hlines(min_mu[i], 0, n_iter[i]+1, colors=standard_colors[2], linestyles='solid')
                axs[fig_count % 2].set(ylabel=mu_names[i]+' updates')

                fig_count += 1

            #   Kernel Parameters
            for i, j in itertools.product(range(d), range(d)):
                for ix_param in range(len(ker_param_names[i][j])):
                    #   Parameter
                    axs[fig_count % 2].plot([fit_log.ker[i][j][n][ix_param] for n in range(n_iter[i]+1)], color=standard_colors[0])
                    if true_ker_param is not None:
                        axs[fig_count % 2].hlines(true_ker_param[i][j][ix_param], 0, n_iter[i]+1, colors=standard_colors[1], linestyles='solid')
                    if min_ker_param is not None:
                        axs[fig_count % 2].hlines(min_ker_param[i][j][ix_param], 0, n_iter[i]+1, colors=standard_colors[2], linestyles='solid')
                    axs[fig_count % 2].set(ylabel=ker_param_names[i][j][ix_param]+' updates')

                    fig_count += 1

            axs[0].set(xlabel='Iteration')
            axs[1].set(xlabel='Iteration')

            if "pad" in kwargs:
                pad = kwargs.get('pad', 300)
                fig.tight_layout(pad=pad)

    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    return axs


def plot_solver_path_contour(x_updates, y_updates, loss_function, scale=1.,
                             axis_resolution=100, x_name='$\u03BC$',
                             y_name='$\u03C9$', true_x=None, true_y=None,
                             min_x=None, min_y=None, dpi=300,
                             contour_color='black', cmap='magma',
                             path_color='whitesmoke',
                             true_param_color='aquamarine',
                             min_param_color='deepskyblue', save=False,
                             filename='image.png', show=False):
    """
    Generic function to plot a contour map of the LSE and the solver path of
    a fitted MHP model. This function applies only when the fitted model and
    the ground truth belong to the same parametric space, and when that
    parametric space is the space of 1-dimensional MHP with only one kernel
    parameter.

    Parameters
    ----------
    x_updates : `list` of `str`, shape=(d,d, )
        Names of kernel parameters to label the y axis.

    y_updates : `list` of `str`, shape=(d, )
        Names of background rates to label the y axis.

    loss_function : `list` of `str`, shape=(d,d, )
        Names of kernel parameters to label the y axis.

    scale : `list` of `float`, shape=(d, ), default=`None`
        True vector of background rates.

    axis_resolution : `list` of `float`, shape=(d,d, ), default=100
        True tensor of kernel parameters.
        This only makes sense if the fitted MHP model belongs to the same
        parametric class as the ground truth MHP.

    x_name : `list` of `float`, shape=(d, ), default=`None`
        The vector of background rates which minimizes the LSE.

    y_name : `list` of `float`, shape=(d,d, ), default=`None`
        The tensor of kernel parameters which minimizes the LSE.
        This only makes sense if the fitted MHP model belongs to the same
        parametric class as the ground truth MHP.

    true_x : `float`, default=None
        True value of the x component of the ground truth MHP.

    true_y : `float`, default=None
        True value of the y component of the ground truth MHP.

    min_x : `float`, default=None
        Value of the x component the minimizer of loss_function.

    min_y : `float`, default=None
        Value of the y component the minimizer of loss_function.

    dpi : `int`, default=300

    contour_color : `str`, default='black'

    cmap : `str`, default='magma'

    path_color : `str`, default='whitesmoke'

    true_param_color : `str`, default='aquamarine'

    min_param_color : `str`, default='deepskyblue'

    save : `bool`, default=`False`
        if `True`, save the plot.

    filename : `str`, default=`'image.png'`

    show : `bool`, default=`False`
        if `True`, show the plot.

    """
    # Define the grid
    if true_x is not None and true_y is not None:
        x_min = min(min(x_updates), true_x)
        x_max = max(max(x_updates), true_x)
        y_min = min(min(y_updates), true_y)
        y_max = max(max(y_updates), true_y)
    else:
        x_min = min(x_updates)
        x_max = max(x_updates)
        y_min = min(y_updates)
        y_max = max(y_updates)

    x_min = x_min*scale-0.2
    x_max = x_max*scale+0.2
    y_min = y_min*scale-0.2
    y_max = y_max*scale+0.2

    x = np.arange(x_min, x_max,
                  (x_max-x_min)/float(axis_resolution))
    y = np.arange(y_min, y_max,
                  (y_max-y_min)/float(axis_resolution))
    X, Y = np.meshgrid(x, y)
    zs = np.array(loss_function([np.ravel(X), np.ravel(Y)]))
    Z = zs.reshape(X.shape)

    fig = plt.figure(0, dpi=dpi)
    ax = fig.add_subplot(111)
    contours = ax.contour(X, Y, Z, 3, colors=contour_color)
    ax.clabel(contours, inline=True, fontsize=8)
    im = ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower',
                   cmap=cmap, alpha=0.9)

    # Initial point
    plt.plot([x_updates[0]], [y_updates[0]], 'o', color=path_color, alpha=1.)

    # Sequence of updates
    plt.plot(x_updates, y_updates, '-', linewidth=2, color=path_color,
             alpha=1.)

    # Reference points
    if true_x is not None and true_y is not None:
        plt.plot([true_x], [true_y], 'o', color=true_param_color, alpha=1.)
    if min_x is not None and min_y is not None:
        plt.plot([min_x], [min_y], 'o', color=min_param_color, alpha=1.)

    # Annotations
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02,
                        ax.get_position().height])
    fig.colorbar(im, cax=cax)

    if save:
        plt.savefig(filename)
    if show:
        fig.show()

    return fig


def plot_kernels(phi, kernel_param, t_min=0., t_max=10., n_samples=10**3,
                 index_from_one=False, log_scale=False, axs=None,
                 save=False, filename='image.png', show=False, **kwargs):
    """
    Generic function to plot the matrix of kernels of an MHP.

    Parameters
    ----------

    """

    d = len(phi)
    if axs is None:
        fig, axs = plt.subplots(d, d, sharex=True, sharey=False, **kwargs)
    x_phi = np.linspace(t_min, t_max, n_samples)
    if d > 1:
        for i, j in itertools.product(range(d), range(d)):
            y_phi = phi[i][j](x_phi, kernel_param[i][j])
            axs[i, j].plot(x_phi, y_phi, color='steelblue')
            axs[i, j].set(ylabel=r'$\phi_{'+str(i+int(index_from_one))+','
                                 + str(j + int(index_from_one)) + '}(t)$')
        axs[d-1, 0].set(xlabel=r'$t$')
        axs[d-1, 1].set(xlabel=r'$t$')

    else:
        y_phi = phi[0][0](x_phi, kernel_param[0][0])
        axs.plot(x_phi, y_phi, color='steelblue')
        axs.set(ylabel=r'$\phi(t)$')
        axs.set(xlabel=r'$t$')
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    return axs


def plot_function_shaded_error(x_vals, list_y_vals, list_y_std,
                               list_colors=standard_colors, xname=None,
                               yname=None, list_labels=[], show=True,
                               labeling=True, scale='lin', **kwargs):
    fig = plt.figure(figsize=(8, 6), **kwargs)
    n_plots = len(list_y_vals)

    for t in range(n_plots):
        if labeling:
            fig.plot(x_vals, list_y_vals[t], color=list_colors[t],
                     label=list_labels[t])
            fig.fill_between(x_vals, list_y_vals[t]-list_y_std[t],
                             list_y_vals[t]+list_y_std[t],
                             color=list_colors[t], alpha=.3)
        else:
            fig.plot(x_vals, list_y_vals[t], color=list_colors[t])
            fig.fill_between(x_vals, list_y_vals[t]-list_y_std[t],
                             list_y_vals[t]+list_y_std[t],
                             color=list_colors[t], alpha=.3)
    if labeling:
        fig.legend()
    if scale == 'log':
        fig.xscale("log")
    if xname is not None:
        fig.xlabel('T')
    if yname is not None:
        fig.ylabel(yname)
    if show:
        fig.show()
    return fig


def animate_plot_sequence(ref_x, list_y, x_min=None, x_max=None, y_min=None,
                          y_max=None, list_y_ref=None, interval=20,
                          save=False, filename='video', show=False, **kwargs):
    """
    Generic function to construct an animation for a sequence of functions.
    Based on
    https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

    Parameters
    ----------

    """
    n_frames = len(list_y)

    if x_min is None:
        x_min = min(ref_x)
    if x_max is None:
        x_max = max(ref_x)
    if y_min is None:
        y_min = min([min(L) for L in list_y])
    if y_max is None:
        y_max = max([max(L) for L in list_y])

    fig = plt.figure(**kwargs)
    ax = plt.axes(xlim=(x_min-10**-5*abs(x_min), x_max+10**-5*abs(x_max)),
                  ylim=(y_min, y_max))
    line, = ax.plot([], [], lw=2, color='steelblue')
    if list_y_ref is not None:
        line2, = ax.plot([], [], lw=2, color='orange')

    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    if list_y_ref is None:
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            x = ref_x
            y = list_y[i]
            line.set_data(x, y)
            time_text.set_text('Step = '+str(i))
            return line, time_text
    else:
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            x = ref_x
            y = list_y[i]
            y_ref = list_y_ref[i]
            line.set_data(x, y)
            line2.set_data(x, y_ref)
            time_text.set_text('Step = '+str(i))
            return line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=interval,
                                   blit=True)
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save:
        anim.save(filename+'.mp4', fps=30,
                  extra_args=['-vcodec', 'libx264'], **kwargs)
    if show:
        plt.show()
    return anim


def annotate_subplots(axes, row_names=None, col_names=None, row_pad=1,
                      col_pad=5, rotate_row_headers=True, **text_kwargs):
    # Based on https://stackoverflow.com/a/25814386
    n_rows, n_cols = axes.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            sbs = ax.get_subplotspec()

            # Putting headers on cols
            if (col_names is not None) and (i == 0):
                ax.annotate(
                    col_names[sbs.colspan.start],
                    xy=(0.5, 1),
                    xytext=(0, col_pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    **text_kwargs,
                )

            # Putting headers on rows
            if (row_names is not None) and (j == 0):
                ax.annotate(
                    row_names[sbs.rowspan.start],
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - row_pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    rotation=rotate_row_headers * 90,
                    **text_kwargs,
                )


# =============================================================================
# Heatmaps
# =============================================================================
def make_heatmap(data, row_labels, col_labels, ax=None, cbar_kw={},
                 cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Based on
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Based on
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_adjacency_matrix(adjacency_matrix, event_names=None,
                          index_from_one=False, annotate=False,
                          cmap="Blues", save=False,
                          filename='image.png', show=True, **kwargs):
    """
    Generic function to plot an adjacency matrix.

    Parameters
    ----------


    """
    d = len(adjacency_matrix)
    if event_names is None:
        event_names = [str(i+int(index_from_one)) for i in range(d)]
    fig, ax = plt.subplots(**kwargs)
    row_labels = [u"\u2192" + ' ' + ev for ev in event_names]
    col_labels = [ev + ' ' + u"\u2192" for ev in event_names]
    im, cbar = make_heatmap(np.array(adjacency_matrix), row_labels, col_labels,
                            ax=ax, cmap=cmap, cbarlabel="Kernel $L_1$ norm")
    if annotate:
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

    fig.tight_layout()
    if show:
        fig.show()
    if save:
        plt.savefig(filename)
    return fig, ax


# =============================================================================
# Braces
# =============================================================================
def get_horizontal_range_brace(x_min, x_max, mid=0.75, beta1=50.0, beta2=100.0,
                               height=1, initial_divisions=11,
                               resolution_factor=1.5):
    """
    Generic function to construct a horizontal brace.
    Based on https://www.py4u.net/discuss/194566

    Parameters
    ----------


    """

    x0 = np.array(())
    tmpx = np.linspace(0, 0.5, initial_divisions)

    tmp = (beta1**2 * (np.exp(beta1*tmpx)) * (1-np.exp(beta1*tmpx))
           / np.power((1 + np.exp(beta1 * tmpx)), 3))

    tmp += (beta2**2 * (np.exp(beta2*(tmpx - 0.5)))
            * (1-np.exp(beta2*(tmpx-0.5)))
            / np.power((1+np.exp(beta2*(tmpx-0.5))), 3))

    for i in range(0, len(tmpx)-1):
        t = int(np.ceil(resolution_factor*max(np.abs(tmp[i:i+2]))
                        / float(initial_divisions)))
        x0 = np.append(x0, np.linspace(tmpx[i], tmpx[i+1], t))
    x0 = np.sort(np.unique(x0))
    # Half brace using sum of two logistic functions
    y0 = mid*2*((1/(1.+np.exp(-1*beta1*x0)))-0.5)
    y0 += (1-mid)*2*(1/(1.+np.exp(-1*beta2*(x0-0.5))))
    # Concatenate and scale x
    x = np.concatenate((x0, 1-x0[::-1])) * float((x_max-x_min)) + x_min
    y = np.concatenate((y0, y0[::-1])) * float(height)
    return (x, y)


def get_vertical_range_brace(y_min, y_max, x_min=0., mid=0.75, beta1=50.0,
                             beta2=100.0, height=1, initial_divisions=11,
                             resolution_factor=1.5):
    """
    Generic function to construct a vertical brace.

    Parameters
    ----------


    """

    (x, y) = get_horizontal_range_brace(y_max, y_min, mid=mid, beta1=beta1,
                                        beta2=beta2, height=height,
                                        initial_divisions=initial_divisions,
                                        resolution_factor=resolution_factor)
    y = x_min+y
    return (y, x)
