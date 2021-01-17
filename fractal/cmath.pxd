cdef:
    struct Plane:
        double xmin, ymin, xmax, ymax

    # [[ Zr, Zi ]
    #  [ Cr, Ci ]]
    struct Point4:
        double zr, zi, cr, ci

    # TODO: Combine Plane/Point4, no reason to be separate types anymore
    # These convert a python list of four floating point values to the pure Cython struct equivalents
    # Convenience as non-cython code cannot instantiate the structs directly
    cdef Plane c_plane(plane)
    cdef Point4 c_point(point)
    cdef Point4 make_p4(double zr, double zi, double cr, double ci) nogil

    cdef Point4 p4_dot(Point4 a, Point4 b) nogil
    cdef Point4 p4_scalar_mult(Point4 a, double scalar) nogil
    cdef Point4 p4_scalar_div(Point4 a, double divisor) nogil
    cdef Point4 p4_add(Point4 a, Point4 b) nogil
    cdef Point4 p4_sub(Point4 a, Point4 b) nogil

    class RenderWindow:
        cdef:
            int resolution
            Plane plane
            double dx, dy

            int x2column(self, double x) nogil
            int y2row(self, double y) nogil
            double col2x(self, int x) nogil
            double row2y(self, int y) nogil

    class RenderConfig:
        """
        m_min, m_max => start and stop points in 4D to iterate trace points over
        m_diff       => m_max - m_min
        m_dt         => Length of chunk along each axis
                        m_dt = m_diff / sqrt(chunks)
        """
        cdef:
            RenderWindow rwin
            int iteration_limit
            int workers
            Point4 m_min, m_max, m_diff, m_dt