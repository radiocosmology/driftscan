import healpy

import matplotlib
from matplotlib import pyplot as plt

def mollview_polfeed(maps, fignum = 1, trans = lambda x: x, feedlabel = ['$XX$', '$XY$', '$YY$'] , pollabel =['$I$', '$Q$', '$U$'], **kwargs):

    for i in range(3):
        for j in range(3):
            healpy.mollview(trans(maps[i][j]), fig=fignum, title = feedlabel[i] + ' - ' + pollabel[j], sub = [3,3,3*i+j+1], **kwargs)

def imshow_polfeed(ims, fignum = 1, trans = lambda x: x, xlabel = None, ylabel = None, feedlabel = ['$XX$', '$XY$', '$YY$'] , pollabel =['$I$', '$Q$', '$U$'] , **kwargs):
    f = plt.figure(fignum)

    for i in range(3):
        for j in range(3):
            ax = f.add_subplot(3, 3, 3*i+j+1)
            ax.imshow(trans(ims[i][j]), **kwargs)
            ax.set_title(feedlabel[i] + ' - ' + pollabel[j])
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)

