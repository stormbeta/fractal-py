cimport numpy as np

cdef:
    struct Plane:
        double xmin, ymin, xmax, ymax

    # [[ Zr, Zi ]
    #  [ Cr, Ci ]]
    struct Point4:
        double zr, zi, cr, ci

    class RenderWindow:
        cdef:
            int resolution
            Plane plane
            double dx, dy

            int x2column(self, double x)
            int y2row(self, double y)
            double col2x(self, int x)
            double row2y(self, int y)

    class RenderConfig:
        cdef:
            RenderWindow rwin
            int iteration_limit
            Point4 m_min, m_max, m_diff, m_dt

    np.ndarray[np.uint8_t, ndim=2] render_histogram(RenderConfig rconfig, int min_density, int max_density)