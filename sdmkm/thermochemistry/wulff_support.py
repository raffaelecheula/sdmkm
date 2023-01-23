################################################################################
# Raffaele Cheula, cheula.raffaele@gmail.com
################################################################################

from pymatgen.core.structure import Structure
import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull
import logging
from collections import OrderedDict
from pymatgen.analysis.wulff import (
    hkl_tuple_to_str,
    get_tri_area,
    WulffFacet,
    WulffShape,
)

logger = logging.getLogger(__name__)

class WulffShapeSupport(WulffShape):
    """
    Generate Wulff Shape from list of miller index and surface energies,
    with given conventional unit cell.
    surface energy (Jm^2) is the length of normal.

    Wulff shape is the convex hull.
    Based on:
    http://scipy.github.io/devdocs/generated/scipy.spatial.ConvexHull.html

    Process:
        1. get wulff simplices
        2. label with color
        3. get wulff_area and other properties

    .. attribute:: debug (bool)

    .. attribute:: alpha
        transparency

    .. attribute:: color_set

    .. attribute:: grid_off (bool)

    .. attribute:: axis_off (bool)

    .. attribute:: show_area

    .. attribute:: off_color
        color of facets off wulff

    .. attribute:: structure
        Structure object, input conventional unit cell (with H ) from lattice

    .. attribute:: miller_list
        list of input miller index, for hcp in the form of hkil

    .. attribute:: hkl_list
        modify hkill to hkl, in the same order with input_miller

    .. attribute:: e_surf_list
        list of input surface energies, in the same order with input_miller

    .. attribute:: lattice
        Lattice object, the input lattice for the conventional unit cell

    .. attribute:: facets
        [WulffFacet] for all facets considering symm

    .. attribute:: dual_cv_simp
        simplices from the dual convex hull (dual_pt)

    .. attribute:: wulff_pt_list

    .. attribute:: wulff_cv_simp
        simplices from the convex hull of wulff_pt_list

    .. attribute:: on_wulff
        list for all input_miller, True is on wulff.

    .. attribute:: color_area
        list for all input_miller, total area on wulff, off_wulff = 0.

    .. attribute:: miller_area
        ($hkl$): area for all input_miller
        
    .. attribute:: miller_supp
        miller index of the facet in contact with the support

    .. attribute:: e_surf_supp
        adhesion energy between the particle and the support

    """

    def __init__(self, lattice, miller_list, e_surf_list, miller_supp=None,
                 e_surf_supp=None, symprec=1e-5):
        """
        Args:
            lattice: Lattice object of the conventional unit cell
            miller_list ([(hkl), ...]: list of hkl or hkil for hcp
            e_surf_list ([float]): list of corresponding surface energies
            miller_supp: hkl of the facet in contact with the support
            e_surf_supp: adhesion energy between particle and support
            symprec (float): for recp_operation, default is 1e-5.
        """
        if miller_supp is None or e_surf_supp is None:
            miller_supp = None
            e_surf_supp = None
            self.n_supp = 0
        else:
            self.n_supp = 1
        
        self.color_ind = list(range(len(miller_list)+self.n_supp))

        self.input_miller_fig = [hkl_tuple_to_str(x) for x in miller_list]
        
        # store input data
        self.structure = Structure(lattice, ["H"], [[0, 0, 0]])
        self.miller_list = tuple([tuple(x) for x in miller_list])
        self.hkl_list = tuple([(x[0], x[1], x[-1]) for x in miller_list])
        self.e_surf_list = tuple(e_surf_list)
        self.miller_supp = miller_supp
        self.e_surf_supp = e_surf_supp
        self.lattice = lattice
        self.symprec = symprec

        if self.miller_supp is not None:
            self.input_miller_fig.append('(support)')
            self.normal_supp = self.miller_supp/sp.linalg.norm(self.miller_supp)

        # 2. get all the data for wulff construction
        # get all the surface normal from get_all_miller_e()
        self.facets = self._get_all_miller_e()
        logger.debug(len(self.facets))

        # 3. consider the dual condition
        dual_pts = [x.dual_pt for x in self.facets]
        dual_convex = ConvexHull(dual_pts)
        dual_cv_simp = dual_convex.simplices
        # simplices	(ndarray of ints, shape (nfacet, ndim))
        # list of [i, j, k] , ndim = 3
        # i, j, k: ind for normal_e_m
        # recalculate the dual of dual, get the wulff shape.
        # conner <-> surface
        # get cross point from the simplices of the dual convex hull
        wulff_pt_list = [self._get_cross_pt_dual_simp(dual_simp)
                         for dual_simp in dual_cv_simp]

        wulff_convex = ConvexHull(wulff_pt_list)
        wulff_cv_simp = wulff_convex.simplices
        logger.debug(", ".join([str(len(x)) for x in wulff_cv_simp]))

        # store simplices and convex
        self.dual_cv_simp = dual_cv_simp
        self.wulff_pt_list = wulff_pt_list
        self.wulff_cv_simp = wulff_cv_simp
        self.wulff_convex = wulff_convex

        self.on_wulff, self.color_area = self._get_simpx_plane()
        
        miller_area = []
        for m, in_mill_fig in enumerate(self.input_miller_fig):
            miller_area.append(
                in_mill_fig + ' : ' + str(round(self.color_area[m], 4)))
        self.miller_area = miller_area

    def _get_all_miller_e(self):
        """
        from self:
            get miller_list(unique_miller), e_surf_list and symmetry
            operations(symmops) according to lattice
        apply symmops to get all the miller index, then get normal,
        get all the facets functions for wulff shape calculation:
            |normal| = 1, e_surf is plane's distance to (0, 0, 0),
            normal[0]x + normal[1]y + normal[2]z = e_surf

        return:
            [WulffFacet]
        """
        all_hkl = []
        color_ind = self.color_ind
        planes = []
        recp = self.structure.lattice.reciprocal_lattice_crystallographic
        #recp_symmops = get_recp_symmetry_operation(self.structure, self.symprec)
        recp_symmops = self.lattice.get_recp_symmetry_operation(self.symprec)

        for i, (hkl, energy) in enumerate(zip(self.hkl_list,
                                              self.e_surf_list)):
            for op in recp_symmops:
                miller = tuple([int(x) for x in op.operate(hkl)])
                if miller not in all_hkl:
                    all_hkl.append(miller)
                    normal = recp.get_cartesian_coords(miller)
                    normal /= sp.linalg.norm(normal)
                    i_new = i
                    energy_new = energy
                    if (self.miller_supp is not None
                        and np.allclose(normal, self.normal_supp)):
                        energy_new = self.e_surf_supp
                        i_new = len(self.miller_list)
                    normal_pt = [x * energy_new for x in normal]
                    dual_pt = [x / energy_new for x in normal]
                    color_plane = color_ind[divmod(i_new, len(color_ind))[1]]
                    planes.append(WulffFacet(normal, energy_new, normal_pt,
                                             dual_pt, color_plane, i_new, hkl))

        # sort by e_surf
        planes.sort(key=lambda x: x.e_surf)
        return planes

    def _get_simpx_plane(self):
        """
        Locate the plane for simpx of on wulff_cv, by comparing the center of
        the simpx triangle with the plane functions.
        """
        on_wulff = [False] * (len(self.miller_list)+self.n_supp)
        surface_area = [0.0] * (len(self.miller_list)+self.n_supp)
        for simpx in self.wulff_cv_simp:
            pts = [self.wulff_pt_list[simpx[i]] for i in range(3)]
            center = np.sum(pts, 0) / 3.0
            # check whether the center of the simplices is on one plane
            for plane in self.facets:
                abs_diff = abs(np.dot(plane.normal, center) - plane.e_surf)
                if abs_diff < 1e-5:
                    on_wulff[plane.index] = True
                    surface_area[plane.index] += get_tri_area(pts)

                    plane.points.append(pts)
                    plane.outer_lines.append([simpx[0], simpx[1]])
                    plane.outer_lines.append([simpx[1], simpx[2]])
                    plane.outer_lines.append([simpx[0], simpx[2]])
                    # already find the plane, move to the next simplices
                    break
        for plane in self.facets:
            plane.outer_lines.sort()
            plane.outer_lines = [line for line in plane.outer_lines
                                 if plane.outer_lines.count(line) != 2]
        return on_wulff, surface_area

    def _get_colors(self, color_set, alpha, off_color, custom_colors={}):
        """
        assign colors according to the surface energies of on_wulff facets.

        return:
            (color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff,
            e_surf_on_wulff_list)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        color_list = [off_color] * (len(self.hkl_list)+self.n_supp)
        color_proxy_on_wulff = []
        miller_on_wulff = []
        e_surf_on_wulff = [(i, e_surf)
                           for i, e_surf in enumerate(self.e_surf_list)
                           if self.on_wulff[i]]

        c_map = plt.get_cmap(color_set)
        e_surf_on_wulff.sort(key=lambda x: x[1], reverse=False)
        e_surf_on_wulff_list = [x[1] for x in e_surf_on_wulff]
        if len(e_surf_on_wulff) > 1:
            cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list),
                                         vmax=max(e_surf_on_wulff_list))
        else:
            # if there is only one hkl on wulff, choose the color of the median
            cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list) - 0.1,
                                         vmax=max(e_surf_on_wulff_list) + 0.1)
        scalar_map = mpl.cm.ScalarMappable(norm=cnorm, cmap=c_map)

        for i, e_surf in e_surf_on_wulff:
            color_list[i] = scalar_map.to_rgba(e_surf, alpha=alpha)
            if tuple(self.miller_list[i]) in custom_colors.keys():
                color_list[i] = custom_colors[tuple(self.miller_list[i])]
            color_proxy_on_wulff.append(
                plt.Rectangle((2, 2), 1, 1, fc=color_list[i], alpha=alpha))
            miller_on_wulff.append(self.input_miller_fig[i])
        scalar_map.set_array([x[1] for x in e_surf_on_wulff])
        color_proxy = [plt.Rectangle((2, 2), 1, 1, fc=x, alpha=alpha)
                       for x in color_list]

        return color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff, \
            e_surf_on_wulff_list

    def get_plot(self, color_set='PuBu', grid_off=True, axis_off=True,
                 show_area=False, alpha=1, off_color='red', direction=None,
                 bar_pos=(0.75, 0.15, 0.05, 0.65), bar_on=False,
                 legend_on=True, aspect_ratio=(8, 8), custom_colors={},
                 azim = None, elev = None, fig = None, sub = None):
        """
        Get the Wulff shape plot.

        Args:
            color_set: default is 'PuBu'
            grid_off (bool): default is True
            axis_off (bool): default is Ture
            show_area (bool): default is False
            alpha (float): chosen from 0 to 1 (float), default is 1
            off_color: Default color for facets not present on the Wulff shape.
            direction: default is (1, 1, 1)
            bar_pos: default is [0.75, 0.15, 0.05, 0.65]
            bar_on (bool): default is False
            legend_on (bool): default is True
            aspect_ratio: default is (8, 8)
            custom_colors ({(h,k,l}: [r,g,b,alpha}): Customize color of each
                facet with a dictionary. The key is the corresponding Miller
                index and value is the color. Undefined facets will use default
                color site. Note: If you decide to set your own colors, it
                probably won't make any sense to have the color bar on.

        Return:
            (matplotlib.pyplot)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d as mpl3
        color_list, color_proxy, color_proxy_on_wulff, \
            miller_on_wulff, e_surf_on_wulff = self._get_colors(
                color_set, alpha, off_color, custom_colors=custom_colors)

        if not direction:
            # If direction is not specified, use the miller indices of
            # maximum area.
            direction = max(self.area_fraction_dict.items(),
                            key=lambda x: x[1])[0]

        if fig is None:
            fig = plt.figure()
            fig.set_size_inches(aspect_ratio[0], aspect_ratio[1])
        if sub is not None:
            plt.subplot(sub)
        if azim is None or elev is None:
            azim, elev = self._get_azimuth_elev([direction[0], direction[1],
                                                direction[-1]])

        wulff_pt_list = self.wulff_pt_list

        ax = mpl3.Axes3D(fig, azim=azim, elev=elev)

        for plane in self.facets:
            # check whether [pts] is empty
            if len(plane.points) < 1:
                # empty, plane is not on_wulff.
                continue
            # assign the color for on_wulff facets according to its
            # index and the color_list for on_wulff
            plane_color = color_list[plane.index]
            lines = list(plane.outer_lines)
            pt = []
            prev = None
            while len(lines) > 0:
                if prev is None:
                    l = lines.pop(0)
                else:
                    for i, l in enumerate(lines):
                        if prev in l:
                            l = lines.pop(i)
                            if l[1] == prev:
                                l.reverse()
                            break
                # make sure the lines are connected one by one.
                # find the way covering all pts and facets
                pt.append(self.wulff_pt_list[l[0]].tolist())
                pt.append(self.wulff_pt_list[l[1]].tolist())
                prev = l[1]
            # plot from the sorted pts from [simpx]
            tri = mpl3.art3d.Poly3DCollection([pt])
            if (self.miller_supp is not None
                and np.allclose(plane.normal, self.normal_supp)):
                tri.set_alpha(0.)
            tri.set_color(plane_color)
            #tri.set_edgecolor("#808080")
            tri.set_edgecolor("#000000")
            ax.add_collection3d(tri)

        # set ranges of x, y, z
        # find the largest distance between on_wulff pts and the origin,
        # to ensure complete and consistent display for all directions
        r_range = max([np.linalg.norm(x) for x in wulff_pt_list])
        ax.set_xlim([-r_range * 1.1, r_range * 1.1])
        ax.set_ylim([-r_range * 1.1, r_range * 1.1])
        ax.set_zlim([-r_range * 1.1, r_range * 1.1])
        # add legend
        if legend_on:
            color_proxy = color_proxy
            if show_area:
                ax.legend(color_proxy, self.miller_area, loc='upper left',
                          bbox_to_anchor=(0, 1), fancybox=True, shadow=False)
            else:
                ax.legend(color_proxy_on_wulff, miller_on_wulff,
                          loc='upper center',
                          bbox_to_anchor=(0.5, 1), ncol=3, fancybox=True,
                          shadow=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Add colorbar
        if bar_on:
            cmap = plt.get_cmap(color_set)
            cmap.set_over('0.25')
            cmap.set_under('0.75')
            bounds = [round(e, 2) for e in e_surf_on_wulff]
            bounds.append(1.2 * bounds[-1])
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # display surface energies
            ax1 = fig.add_axes(bar_pos)
            cbar = mpl.colorbar.ColorbarBase(
                ax1, cmap=cmap, norm=norm, boundaries=[0] + bounds + [10],
                extend='both', ticks=bounds[:-1], spacing='proportional',
                orientation='vertical')
            cbar.set_label('Surface Energies ($J/m^2$)', fontsize=100)

        if grid_off:
            ax.grid(b=False)
        if axis_off:
            ax.axis('off')
        return plt

    @property
    def miller_area_dict(self):
        """
        Returns {hkl: area_hkl on wulff}
        """
        area_dict = OrderedDict(zip(self.miller_list, self.color_area))
        if self.miller_supp is not None:
            area_dict['(support)'] = self.color_area[-1]
        return area_dict

################################################################################
# END
################################################################################