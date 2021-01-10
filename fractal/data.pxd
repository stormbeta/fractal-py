cdef:
    struct Plane:
        double xmin, ymin, xmax, ymax

    cdef Plane c_plane(plane)

    # [[ Zr, Zi ]
    #  [ Cr, Ci ]]
    struct Point4:
        double zr, zi, cr, ci

    cdef Point4 p4_dot(Point4 a, Point4 b)
    cdef Point4 p4_scalar_mult(Point4 a, double scalar)
    cdef Point4 p4_scalar_div(Point4 a, double divisor)
    cdef Point4 p4_add(Point4 a, Point4 b)
    cdef Point4 p4_sub(Point4 a, Point4 b)
    cdef Point4 p4_iterate(Point4 a)

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
            int workers
            Point4 m_min, m_max, m_diff, m_dt