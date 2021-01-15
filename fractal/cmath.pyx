from math import *
from .iterator cimport p4_iterate

cdef:
    # Yes I know numpy already implements all these, but the overhead for tiny 2x2 matrices is large, numpy is meant for larger datasets
    # Dot-product
    # [[ Zr, Zi ]   \/  [[ Zr, Zi ]
    #  [ Cr, Ci ]]  /\   [ Cr, Ci ]]
    cdef Point4 p4_dot(Point4 a, Point4 b):
        return Point4(a.zr * b.zr + a.zi * b.cr, a.zr * b.zi + a.zi * b.ci,
                      a.cr * b.zr + a.ci * b.cr, a.cr * b.zi + a.ci * b.ci)

    cdef Point4 p4_scalar_mult(Point4 a, double scalar):
        return Point4(a.zr * scalar, a.zi * scalar,
                      a.cr * scalar, a.ci * scalar)

    cdef Point4 p4_scalar_div(Point4 a, double divisor):
        return Point4(a.zr / divisor, a.zi / divisor,
                      a.cr / divisor, a.ci / divisor)

    cdef Point4 p4_add(Point4 a, Point4 b):
        return Point4(a.zr + b.zr, a.zi + b.zi,
                      a.cr + b.cr, a.ci + b.ci)

    cdef Point4 p4_sub(Point4 a, Point4 b):
        return Point4(a.zr - b.zr, a.zi - b.zi,
                      a.cr - b.cr, a.ci - b.ci)

    cdef Point4 p4_square_z(Point4 a):
        return Point4(a.zr * a.zr - a.zi * a.zi, 2 * a.zr * a.zi, a.cr, a.ci)

    cdef Point4 p4_cube_z(Point4 a):
        return Point4(a.zr * a.zr * a.zr - 6*a.zr*a.zi*a.zi,
                      6*a.zr*a.zr*a.zi - a.zi*a.zi*a.zi,
                      a.cr, a.ci)

    # TODO: IDEA: what we mixed multiple iterators into the same render?
    #       E.g. iterations 0-100: func A, iterations 101-1000, func B

    cdef Plane c_plane(plane):
        return Plane(plane[0], plane[1], plane[2], plane[3])

    cdef Point4 c_point(point):
        return Point4(point[0], point[1], point[2], point[3])

    class RenderWindow:
        def __cinit__(self, Plane plane, int resolution):
            self.resolution = resolution
            self.plane = plane
            self.dx = ((plane.xmax - plane.xmin) / resolution)
            self.dy = ((plane.ymax - plane.ymin) / resolution)

        # NOTE: Inlining these doesn't seem to actually help performance
        cdef int x2column(self, double x):
            return <int>(((x - self.plane.xmin) / self.dx) - 1)

        cdef int y2row(self, double y):
            return <int>(self.resolution - int((y - self.plane.ymin) / self.dy) - 1)

        cdef double col2x(self, int x):
            return self.plane.xmin + (<double>x * self.dx)

        cdef double row2y(self, int y):
            return self.plane.ymax - (<double>y * self.dy)


    class RenderConfig:
        def __cinit__(self, RenderWindow rwin, int iteration_limit,
                      Point4 m_min, Point4 m_max):
            self.rwin = rwin
            self.iteration_limit = iteration_limit
            plane = rwin.plane
            self.m_min = m_min
            self.m_max = m_max
            self.m_diff = p4_sub(self.m_max, self.m_min)
            self.m_dt = p4_scalar_div(self.m_diff, rwin.resolution)
