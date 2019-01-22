


def semseg2mesh(SuperSegmentationObject sso, semseg_key, int nb_views=None, dest_path=None, k=1, float[:,:]colors):
    ### What's the type of i_views? It's a table of sth
    colors = colors * 255

    if nb_views is None:
        # load default
        i_views = sso.load_views("index").flatten()
    else:
        # load special views
        i_views = sso.load_views("index{}".format(nb_views)).flatten()
    spiness_views = sso.load_views(semseg_key).flatten()
    cdef  int[:] ind = sso.mesh[0]
