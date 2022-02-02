from warp import picmi
import numpy as np


class RectChamber:

    def __init__(self, width, height, z_start, z_end, ghost_x=1e-3,
                 ghost_y=1e-3, ghost_z=1e-3, condid=1):
        print('Using rectangular chamber with xaper: %1.2e, yaper: %1.2e'
              % (width / 2., height / 2.))
        self.width = width
        self.height = height
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -width / 2 - self.ghost_x
        self.xmax = -self.xmin
        self.ymin = -height / 2 - self.ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - self.ghost_z
        self.zmax = z_end + self.ghost_z

        self.lower_bound = [-width / 2, -height / 2, z_start]
        self.upper_bound = [width / 2, height / 2, z_end]

        self.condid = condid

        upper_box = picmi.warp.YPlane(y0=height / 2 - 1.e-10, ysign=1,
                                      condid=condid)
        lower_box = picmi.warp.YPlane(y0=-height / 2 + 1.e-10, ysign=-1,
                                      condid=condid)
        left_box = picmi.warp.XPlane(x0=width / 2 - 1.e-10, xsign=1, condid=condid)
        right_box = picmi.warp.XPlane(x0=-width / 2 + 1.e-10, xsign=-1,
                                      condid=condid)
        self.z_inj_beam = (0.2*self.z_start+0.8*self.zmin)
        self.conductors = upper_box + lower_box + left_box + right_box

    def is_outside(self, xx, yy, zz):
        width = self.width
        height = self.height
        z_start = self.z_start
        z_end = self.z_end
        return np.logical_or.reduce([abs(xx) > width / 2, abs(yy) > height / 2,
                                     zz < z_start, zz > z_end])


class LHCChamber:

    def __init__(self, length, z_start, z_end, ghost_x=1e-3, ghost_y=1e-3,
                 ghost_z=1e-3, condid=1):
        print('Using the LHC chamber')

        self.height = 36e-3
        self.radius = 23e-3
        self.width = self.radius * 2
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -width / 2. - ghost_x
        self.xmax = -self.xmin
        self.ymin = -height / 2 - ghost_y
        self.ymax = -self.ymin
        self.zmin = zs_dipo - ghost_z
        self.zmax = ze_dipo + ghost_z

        self.lower_bound = [-radius, -radius, z_start]
        self.upper_bound = [radius, radius, z_end]

        upper_box = picmi.warp.YPlane(y0=height / 2, ysign=1,
                                      condid=condid)
        lower_box = picmi.warp.YPlane(y0=-height / 2, ysign=-1,
                                      condid=condid)
        pipe = picmi.warp.ZCylinderOut(radius=radius, length=self.length,
                                       condid=condid)

        self.conductors = pipe + upper_box + lower_box

    def is_outside(self, xx, yy, zz):
        r0_sq = np.square(x0) + np.square(y0)
        return np.logical_or.reduce([r0sq > self.radius ** 2,
                                     abs(yy) > self.height,
                                     zz < z_start, zz > z_end])


class CircChamber:

    def __init__(self, radius, z_start, z_end, ghost_x=1e-3, ghost_y=1e-3,
                 ghost_z=1e-3, condid=1):
        print('Using a circular chamber with radius %1.2e' % radius)

        self.radius = radius
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -radius - ghost_x
        self.xmax = -self.xmin
        self.ymin = -radius - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        self.lower_bound = [-radius, -radius, z_start]
        self.upper_bound = [radius, radius, z_end]

        pipe = picmi.warp.ZCylinderOut(radius=radius, length=self.length,
                                       condid=condid)

        self.conductors = pipe

    def is_outside(self, xx, yy, zz):
        r0_sq = np.square(xx) + np.square(yy)
        return np.logical_or.reduce([r0sq > self.radius ** 2,
                                     zz < z_start, zz > z_end])


class EllipChamber:

    def __init__(self, r_x, r_y, z_start, z_end, ghost_x=1e-3, ghost_y=1e-3,
                 ghost_z=1e-3, condid=1):
        print('Using an elliptic chamber chamber with r_x %1.2e and r_y %1.2e' % (r_x, r_y))

        self.r_x = r_x
        self.r_y = r_y
        self.z_start = z_start
        self.z_end = z_end
        self.length = z_end - z_start
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin = -self.r_x - ghost_x
        self.xmax = -self.xmin
        self.ymin = -self.r_y - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        self.lower_bound = [-self.r_x, -self.r_y, z_start]
        self.upper_bound = [self.r_x, self.r_y, z_end]
        self.ellipticity = (self.r_x - self.r_y) / self.r_x
        self.zcent = (z_start + z_end) / 2
        pipe = picmi.warp.ZCylinderEllipticOut(ellipticity=self.ellipticity,
                                               radius=self.r_x,
                                               length=self.length,
                                               z_cent=self.zcent,
                                               condid=condid)

        self.conductors = pipe

    def is_outside(self, xx, yy, zz):
        r0_sq = np.square(xx / self.r_x) + np.square(yy / self.r_y)
        return np.logical_or.reduce([r0sq > 1,
                                     zz < z_start, zz > z_end])


class CrabCavitySquared:

    def __init__(self, z_start, z_end, disp=0, ghost_x=10e-3, ghost_y=10e-3,
                 ghost_z=1e-3, condid=1):
        print('Simulating ECLOUD in a consistent crab cavity')

        self.z_start = z_start
        self.z_end = z_end
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z
        self.z_inj_beam = self.z_start
        
        self.l_main_y = 240e-3 #242e-3
        self.l_main_x = 300e-3
        self.l_main_z = 354e-3 #350e-3
        self.l_beam_pipe = 84e-3 #for y
        self.l_beam_pipe_x = 84e-3
        self.l_int = 60e-3  #62e-3
        self.l_main_int_y = self.l_main_y - self.l_beam_pipe / 2
        self.l_main_int_z = self.l_main_z / 2 - self.l_int
        self.l_main_int_x = self.l_main_x / 2 - self.l_int
        self.disp = disp
        assert z_start < - self.l_main_z / 2, 'z_start must be lower than -175mm'
        assert z_end > self.l_main_z / 2, 'z_end must be higher than 175mm'

        self.xmin = -self.l_main_x/2 - ghost_x
        self.xmax = -self.xmin
        self.ymin = -self.l_main_y / 2 - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        box1 = picmi.warp.Box(zsize=self.zmax - self.zmin,
                              xsize=self.xmax - self.xmin,
                              ysize=self.ymax - self.ymin, condid=condid,
                              zcent=0.5 * (self.zmax + self.zmin))
        box2 = picmi.warp.Box(zsize=self.zmax - self.zmin,
                              xsize=self.l_beam_pipe_x - 2.e-10,
                              ysize=self.l_beam_pipe - 2.e-10, condid=condid,
                              zcent=0.5 * (self.zmax + self.zmin))
        box3 = picmi.warp.Box(zsize=self.l_main_z - 2.e-10,
                              xsize=self.l_main_x - 2.e-10,
                              ysize=self.l_main_y - 2.e-10, condid=condid)

        self.ycen_up = self.l_beam_pipe / 2 + self.l_main_int_y
        self.ycen_down = - self.ycen_up
        box4 = picmi.warp.Box(zsize=2 * self.l_main_int_z + 2.e-10,
                              xsize=2 * self.l_main_int_x + 2.e-10,
                              ysize=2 * self.l_main_int_y + 2.e-10, ycent=self.ycen_up,
                              condid=condid)
        box5 = picmi.warp.Box(zsize=2 * self.l_main_int_z + 2.e-10,
                              xsize=2 * self.l_main_int_x + 2.e-10,
                              ysize=2 * self.l_main_int_y + 2.e-10, ycent=self.ycen_down,
                              condid=condid)

        self.conductors = box1 - box2 - box3 + box4 + box5

        self.upper_bound = [self.l_main_x / 2, self.l_main_y / 2, self.z_end]
        self.lower_bound = [-self.l_main_x / 2, -self.l_main_y / 2, self.z_start]
        # self.upper_bound = [self.l_main_int_x, self.l_main_int_y, self.l_main_int_z]
        # self.lower_bound = [-self.l_main_int_x, -self.l_main_int_y, -self.l_main_int_z]

    def is_outside(self, xx, yy, zz):
        flag_in_box = np.logical_and.reduce([abs(xx) < self.l_main_x / 2,
                                             abs(yy) < self.l_main_y / 2,
                                             abs(zz) < self.l_main_z / 2])

        flag_out_poles = np.logical_and.reduce([abs(xx) < self.l_main_int_x,
                                                abs(zz) < self.l_main_int_z,
                                                abs(yy) > self.l_beam_pipe / 2])
        zs_pipe_left = self.z_start
        ze_pipe_left = -self.l_main_z / 2
        zs_pipe_right = self.l_main_z / 2
        ze_pipe_right = self.z_end
        flag_in_pipe_l = np.logical_and.reduce([abs(xx) < self.l_beam_pipe,
                                                abs(yy) < self.l_beam_pipe,
                                                zz > zs_pipe_left,
                                                zz < ze_pipe_left])
        flag_in_pipe_r = np.logical_and.reduce([abs(xx) < self.l_beam_pipe,
                                                abs(yy) < self.l_beam_pipe,
                                                zz > zs_pipe_right,
                                                zz < ze_pipe_right])

        flag_in_core = np.logical_and.reduce([flag_in_box, np.logical_not(flag_out_poles)])

        return np.logical_not(np.logical_or.reduce([flag_in_core, flag_in_pipe_r,
                                                    flag_in_pipe_l]))


class CrabCavityWaveguide:

    def __init__(self, z_start, z_end, disp=0, ghost_x=10e-3, ghost_y=10e-3,
                 ghost_z=1e-3, condid=1):
        print('Simulating ECLOUD in a consistent crab cavity')

        self.z_start = z_start
        self.z_end = z_end
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z
        self.z_inj_beam = self.z_start
        self.l_main_y = 240e-3 #242e-3
        self.l_main_x = 300e-3
        self.l_main_z = 354e-3 #350e-3
        self.l_beam_pipe = 84e-3 #for y
        self.l_beam_pipe_x = 80e-3
        self.l_int = 60e-3  #62e-3
        self.l_main_int_y = self.l_main_y - self.l_beam_pipe / 2
        self.l_main_int_z = self.l_main_z / 2 - self.l_int
        self.l_main_int_x = self.l_main_x / 2 - self.l_int
        self.disp = disp
        assert z_start < - self.l_main_z / 2, 'z_start must be lower than -175mm'
        assert z_end > self.l_main_z / 2, 'z_end must be higher than 175mm'
        self.y_min_wg = 48e-3 + self.disp    #48.4e-3 + self.disp
        self.y_max_wg = 96e-3 + self.disp    #96.8e-3 + self.disp
        self.x_min_wg_rest = -120e-3
        self.x_max_wg_rest = 120e-3
        self.x_min_wg = -200e-3
        self.x_max_wg = 200e-3
        self.z_rest = -204e-3 #-205e-3

        self.xmin = self.x_min_wg - ghost_x
        self.xmax = -self.xmin
        self.ymin = -self.l_main_y / 2 - ghost_y
        self.ymax = -self.ymin
        self.zmin = z_start - ghost_z
        self.zmax = z_end + ghost_z

        self.z_max_wg = -0.98*self.l_main_z / 2 
        self.z_min_wg = self.zmin*1.2

        box1 = picmi.warp.Box(zsize=self.zmax - self.zmin,
                              xsize=self.xmax - self.xmin,
                              ysize=self.ymax - self.ymin, condid=condid,
                              zcent=0.5 * (self.zmax + self.zmin))
        box2 = picmi.warp.Box(zsize=self.zmax - self.zmin,
                              xsize=self.l_beam_pipe_x - 2.e-10,
                              ysize=self.l_beam_pipe - 2.e-10, condid=condid,
                              zcent=0.5 * (self.zmax + self.zmin))
        box3 = picmi.warp.Box(zsize=self.l_main_z - 2.e-10,
                              xsize=self.l_main_x - 2.e-10,
                              ysize=self.l_main_y - 2.e-10, condid=condid)

        self.ycen_up = self.l_beam_pipe / 2 + self.l_main_int_y
        self.ycen_down = - self.ycen_up
        box4 = picmi.warp.Box(zsize=2 * self.l_main_int_z + 2.e-10,
                              xsize=2 * self.l_main_int_x + 2.e-10,
                              ysize=2 * self.l_main_int_y + 2.e-10, ycent=self.ycen_up,
                              condid=condid)
        box5 = picmi.warp.Box(zsize=2 * self.l_main_int_z + 2.e-10,
                              xsize=2 * self.l_main_int_x + 2.e-10,
                              ysize=2 * self.l_main_int_y + 2.e-10, ycent=self.ycen_down,
                              condid=condid)

        self.ycen6 = 0.5 * (self.y_min_wg + self.y_max_wg)
        self.zcen6 = 0.5 * (self.z_rest + self.z_max_wg)
        box6 = picmi.warp.Box(zsize=self.z_max_wg - self.z_rest - 2.e-10,
                              xsize=self.x_max_wg_rest - self.x_min_wg_rest - 2.e-10,
                              ysize=self.y_max_wg - self.y_min_wg - 2.e-10,
                              ycent=self.ycen6, zcent=self.zcen6)
        self.ycen7 = 0.5 * (self.y_min_wg + self.y_max_wg)
        self.zcen7 = 0.5 * (self.z_min_wg + self.z_rest)
        box7 = picmi.warp.Box(zsize=self.z_rest - self.z_min_wg + 2.e-10,
                              xsize=self.x_max_wg - self.x_min_wg - 2.e-10,
                              ysize=self.y_max_wg - self.y_min_wg - 2.e-10,
                              ycent=self.ycen7, zcent=self.zcen7)

        self.conductors = box1 - box2 - box3 + box4 + box5 - box6 - box7

        self.upper_bound = [self.l_main_x / 2, self.l_main_y / 2, self.l_main_z / 2]
        self.lower_bound = [-self.l_main_x / 2, -self.l_main_y / 2, -self.l_main_z / 2]
        # self.upper_bound = [self.l_main_int_x, self.l_main_int_y, self.l_main_int_z]
        # self.lower_bound = [-self.l_main_int_x, -self.l_main_int_y, -self.l_main_int_z]

    def is_outside(self, xx, yy, zz):
        flag_in_box = np.logical_and.reduce([abs(xx) < self.l_main_x / 2,
                                             abs(yy) < self.l_main_y / 2,
                                             abs(zz) < self.l_main_z / 2])

        flag_out_poles = np.logical_and.reduce([abs(xx) < self.l_main_int_x,
                                                abs(zz) < self.l_main_int_z,
                                                abs(yy) > self.l_beam_pipe / 2])
        zs_pipe_left = self.z_start
        ze_pipe_left = -self.l_main_z / 2
        zs_pipe_right = self.l_main_z / 2
        ze_pipe_right = self.z_end
        flag_in_pipe_l = np.logical_and.reduce([abs(xx) < self.l_beam_pipe,
                                                abs(yy) < self.l_beam_pipe,
                                                zz > zs_pipe_left,
                                                zz < ze_pipe_left])
        flag_in_pipe_r = np.logical_and.reduce([abs(xx) < self.l_beam_pipe,
                                                abs(yy) < self.l_beam_pipe,
                                                zz > zs_pipe_right,
                                                zz < ze_pipe_right])

        flag_in_core = np.logical_and.reduce([flag_in_box, np.logical_not(flag_out_poles)])

        return np.logical_not(np.logical_or.reduce([flag_in_core, flag_in_pipe_r,
                                                    flag_in_pipe_l]))

class CrabCavityRoundCyl:
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, condid=1, nz = 0):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin =  -0.2 - ghost_x
        self.xmax = 0.2 + ghost_x
        self.ymin = -0.2 - ghost_y
        self.ymax = 0.2 + ghost_y
        self.zmin = -0.3 - ghost_z
        self.zmax = 0.3 + ghost_z

        L_int_x = 0.07
        L_int_y = 0.104
        ell = L_int_x/L_int_y
#ell = 1/ell
        h_up = h_down = 0.1

        L_out_z = 0.141/ell
        L_slope_out = 0.02/ell
        L_int_z = 0.07/ell
        L_slope_int = 0.0267/ell
        cylbody = picmi.warp.YCylinderElliptic(ellipticity = ell, radius = L_out_z,
                                              length=0.284/2, ycent = 0.284/4)

        cylup = picmi.warp.YCylinderElliptic(ellipticity = ell, radius = L_int_z,
                                         length=h_up, ycent = 0.5*(0.042 + 0.142))
        cyldown = picmi.warp.YCylinderElliptic(ellipticity = ell, radius = L_int_z,
                                           length=h_down,
                                           ycent = -0.5*(0.042 + 0.142))

        cylbody = picmi.warp.YCylinderElliptic(ellipticity = ell, radius = L_out_z,
                                                length=0.284)

        cyl_l = picmi.warp.ZCylinder(radius=0.042, zlower = 1.05*self.zmin, zupper = -0.18)
        cyl_r = picmi.warp.ZCylinder(radius=0.042, zupper = 1.05*self.zmax, zlower = 0.18)
        box = picmi.warp.Box(xsize=(self.xmax-self.xmin), ysize=(self.ymax-self.ymin), zsize=(self.zmax-self.zmin))
        self.conductors = box - cylbody  + cylup + cyldown - cyl_l - cyl_r
#cone = picmi.warp.XConeElliptic(ellipticity=ell, r_xmin=L_int_x, r_xmax=L_int_x, length=h_up)

        self.lower_bound = [-0.16, -0.15, -0.25]
        self.upper_bound = [0.16, 0.15, 0.25]

        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0
        self.z_inj_beam = self.zmin + dz

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.


class CrabCavityRoundCone:
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, condid=1, nz = 0):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin =  -0.2 - ghost_x
        self.xmax = 0.2 + ghost_x
        self.ymin = -0.2 - ghost_y
        self.ymax = 0.2 + ghost_y
        self.zmin = -0.3 - ghost_z
        self.zmax = 0.3 + ghost_z

        L_int_x = 0.07
        L_int_y = 0.104
        ell = L_int_x/L_int_y
#ell = 1/ell
        h_up = h_down = 0.1

        L_out_z = 0.141/ell
        L_slope_out = 0.02/ell
        L_int_z = 0.07/ell
        L_slope_int = 0.0267/ell
        conebody_up = picmi.warp.YConeElliptic(ellipticity = ell, r_ymin = L_out_z,
                                              r_ymax = L_out_z + L_slope_out,
                                              length=0.284/2, ycent = 0.284/4)
        conebody_down = picmi.warp.YConeElliptic(ellipticity = ell, r_ymax = L_out_z,
                                                r_ymin = L_out_z + L_slope_out,
                                                length=0.284/2, ycent = -0.284/4)

        coneup = picmi.warp.YConeElliptic(ellipticity = ell, r_ymin = L_int_z,
                                         r_ymax = L_int_z + L_slope_int,
                                         length=h_up, ycent = 0.5*(0.042 + 0.142))
        conedown = picmi.warp.YConeElliptic(ellipticity = ell, r_ymax = L_int_z,
                                           r_ymin = L_int_z + L_slope_int, length=h_down,
                                           ycent = -0.5*(0.042 + 0.142))



        box = picmi.warp.Box(xsize=(self.xmax-self.xmin), ysize=(self.ymax-self.ymin), zsize=(self.zmax-self.zmin))

        self.conductors = box - conebody_up - conebody_down  + coneup + conedown - cyl_l - cyl_r

        self.lower_bound = [-0.16, -0.15, -0.25]
        self.upper_bound = [0.16, 0.15, 0.25]

        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0
        self.z_inj_beam = self.zmin + dz

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.

class CrabCavityRound:
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, condid=1, nz = 0, meas_kick=False):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin =  -0.165 - ghost_x
        self.xmax = 0.165 + ghost_x
        self.ymin = -0.150 - ghost_y
        self.ymax = 0.150 + ghost_y
        if not meas_kick:
            self.zmin = -0.300 - ghost_z
            self.zmax = 0.300 + ghost_z
        else:
            self.zmin = -0.300 - ghost_z
            self.zmax = 0.900 + ghost_z

        self.L_int_x = 0.071
        self.L_slope_ext_x = 0.02
        self.L_slope_int_x = 0.0267
        #self.L_int_z = 0.104
        self.L_out_x = 0.0643
        self.L_out_tot_x = self.L_int_x+self.L_slope_int_x+self.L_out_x-self.L_slope_ext_x

        self.ell = 1/1.2 
        #self.ell = self.L_int_x/self.L_int_z
        self.L_int_z = self.L_int_x/self.ell
#ell = 1/ell
        #h_up = h_down = 0.1

        self.L_out_z = self.L_out_tot_x/self.ell
        self.L_slope_ext_z = self.L_slope_ext_x/self.ell
        self.L_slope_int_z = self.L_slope_int_x/self.ell

        self.r_beam_pipe = 0.042
        self.H_out = 0.285

        cyl_l = picmi.warp.ZCylinder(radius=self.r_beam_pipe, zlower = 1.05*self.zmin, zupper = 1.05*self.zmax)
#        cyl_r = picmi.warp.ZCylinder(radius=self.r_beam_pipe, zupper = 1.05*self.zmax, zlower = 0.18)

        rminofzdata = [self.L_int_z+self.L_slope_int_z, self.L_int_z, self.L_int_z, self.L_int_z+self.L_slope_int_z]
        zmindata = [-self.H_out/2, -self.r_beam_pipe, self.r_beam_pipe, self.H_out/2]
        rmaxofzdata = [self.L_out_z+self.L_slope_ext_z, self.L_out_z, self.L_out_z+self.L_slope_ext_z]
        zmaxdata = [-self.H_out/2, 0, self.H_out/2]
        bodyCC_rev = picmi.warp.YSrfrvEllipticInOut(ellipticity=self.ell, rminofydata=rminofzdata, rmaxofydata=rmaxofzdata, ymindata=zmindata, ymaxdata=zmaxdata)
        bodyCC_cyl = picmi.warp.YCylinderElliptic(ellipticity = self.ell, radius=1.01*self.L_int_z, length = 2*self.r_beam_pipe)
        bodyCC= bodyCC_rev + bodyCC_cyl

        box = picmi.warp.Box(xsize=(self.xmax-self.xmin), ysize=(self.ymax-self.ymin), zsize=(self.zmax-self.zmin))
        self.conductors = box - bodyCC - cyl_l #- cyl_r #cylup + cyldown #+ (box - bodyCC)

        #self.lower_bound = [-0.16, -0.15, -0.25]
        #self.upper_bound = [0.16, 0.15, 0.25]
        self.lower_bound = np.array([self.xmin, self.ymin, self.zmin])
        self.upper_bound = np.array([self.xmax, self.ymax, self.zmax])
        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0
        self.z_inj_beam = self.zmin + dz

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.


class SimpleRFD:
    
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, nz = 0, condid=1, meas_kick=False):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.r_body = 0.1
        self.Lz_body = 0.450
        self.L_body = 0.279
        self.r_beam_pipe = 0.042
        self.Lz_pole = 0.353428
        self.Ly_pole = 0.1438274
        self.h_pole = 0.102

        self.xmin =  -0.285/2 - ghost_x
        self.xmax = 0.285/2 + ghost_x
        self.ymin = -0.285/2 - ghost_y
        self.ymax = 0.285/2 + ghost_y
        if not meas_kick:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + ghost_z
        else:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + 1.002 + ghost_z

        self.l_beam_pipe = self.zmax - self.zmin   

        cyl_body_1 = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body, xcent=self.L_body/2-self.r_body, ycent=self.L_body/2-self.r_body)
        cyl_body_2 = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body, xcent=-self.L_body/2+self.r_body, ycent=self.L_body/2-self.r_body)
        cyl_body_3 = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body, xcent=self.L_body/2-self.r_body, ycent=-self.L_body/2+self.r_body)
        cyl_body_4 = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body, xcent=-self.L_body/2+self.r_body, ycent=-self.L_body/2+self.r_body)

        box_body_h = picmi.warp.Box(xsize=self.L_body, ysize=self.L_body-2*self.r_body, zsize=self.Lz_body)
        box_body_v = picmi.warp.Box(xsize=self.L_body-2*self.r_body, ysize=self.L_body, zsize=self.Lz_body)
            
        box_pole_up = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=self.L_body/2-self.h_pole/2)
        box_pole_down = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=-self.L_body/2+self.h_pole/2)

        beam_pipe = picmi.warp.ZCylinder(radius=self.r_beam_pipe, length=self.l_beam_pipe)

        body = -cyl_body_1 - cyl_body_2 - cyl_body_3 - cyl_body_4 - box_body_h - box_body_v

        beam_pipe_left = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=-self.Lz_body/2-self.zmin, zcent=0.5*(-self.Lz_body/2+self.zmin))
        beam_pipe_right = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=self.zmax-self.Lz_body/2, zcent=0.5*(self.Lz_body/2+self.zmax))

        self.conductors = body + box_pole_up + box_pole_down - beam_pipe + beam_pipe_left + beam_pipe_right

        self.lower_bound = np.array([self.xmin, self.ymin, self.zmin])
        self.upper_bound = np.array([self.xmax, self.ymax, self.zmax])

        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0

        self.z_inj_beam = self.zmin + dz
    

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.
 

class SimpleRFDCyl:
    
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, nz = 0, condid=1, meas_kick=False):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.r_body = 0.145
        self.Lz_body = 0.474
        self.L_body = 0.279
        self.r_beam_pipe = 0.042
        self.Lz_pole = 0.353428
        self.Ly_pole = 0.1438274
        self.h_pole = 0.102

        self.xmin =  -0.285/2 - ghost_x
        self.xmax = 0.285/2 + ghost_x
        self.ymin = -0.285/2 - ghost_y
        self.ymax = 0.285/2 + ghost_y
        if not meas_kick:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + ghost_z
        else:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + 1.002 + ghost_z

        self.l_beam_pipe = self.zmax - self.zmin   

        cyl_body = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body)

        box_body = picmi.warp.Box(xsize=self.L_body, ysize=self.L_body, zsize=self.Lz_body)
            
        box_pole_up = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=self.L_body/2-self.h_pole/2)
        box_pole_down = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=-self.L_body/2+self.h_pole/2)

        beam_pipe = picmi.warp.ZCylinder(radius=self.r_beam_pipe, length=self.l_beam_pipe)

        
        body = cyl_body*box_body 

        beam_pipe_left = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=-self.Lz_body/2-self.zmin, zcent=0.5*(-self.Lz_body/2+self.zmin))
        beam_pipe_right = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=self.zmax-self.Lz_body/2, zcent=0.5*(self.Lz_body/2+self.zmax))

        self.conductors = -body + box_pole_up + box_pole_down - beam_pipe + beam_pipe_left + beam_pipe_right

        self.lower_bound = np.array([self.xmin, self.ymin, self.zmin])
        self.upper_bound = np.array([self.xmax, self.ymax, self.zmax])

        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0

        self.z_inj_beam = self.zmin + dz
    

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.
 

class SimpleRFDRot:
    
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, nz = 0, condid=1, meas_kick=False):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.r_body = 0.145
        self.Lz_body = 0.474
        self.L_body = 0.279
        self.r_beam_pipe = 0.042
        self.Lz_pole = 0.334
        self.Ly_pole = 0.1438274
        self.h_pole = 0.101
        self.L1 = 0.03

        self.xmin =  -0.285/2 - ghost_x
        self.xmax = 0.285/2 + ghost_x
        self.ymin = -0.285/2 - ghost_y
        self.ymax = 0.285/2 + ghost_y
        if not meas_kick:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + ghost_z
        else:
            self.zmin = -0.750/2 - ghost_z
            self.zmax = 0.750/2 + 1.002 + ghost_z

        self.l_beam_pipe = self.zmax - self.zmin   

        #cyl_body = picmi.warp.ZCylinder(radius=self.r_body, length=self.Lz_body)

        rsrf = [0, self.r_beam_pipe, self.r_body, self.r_body, self.r_beam_pipe, 0]
        zsrf = [-self.Lz_body/2, -self.Lz_body/2, -self.Lz_body/2+self.L1, 
                self.Lz_body/2-self.L1, self.Lz_body/2, self.Lz_body/2]

        rot_body = picmi.warp.ZSrfrv(rsrf=rsrf, zsrf=zsrf)

        box_body = picmi.warp.Box(xsize=self.L_body, ysize=self.L_body, zsize=self.Lz_body)
            
        box_pole_up = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=self.L_body/2-self.h_pole/2)
        box_pole_down = picmi.warp.Box(xsize=self.h_pole, ysize=self.Ly_pole, zsize=self.Lz_pole, xcent=-self.L_body/2+self.h_pole/2)

        beam_pipe = picmi.warp.ZCylinder(radius=self.r_beam_pipe, length=self.l_beam_pipe)

        
        body = rot_body*box_body 

        beam_pipe_left = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=-self.Lz_body/2-self.zmin, zcent=0.5*(-self.Lz_body/2+self.zmin))
        beam_pipe_right = picmi.warp.ZCylinderOut(radius=self.r_beam_pipe, length=self.zmax-self.Lz_body/2, zcent=0.5*(self.Lz_body/2+self.zmax))

        self.conductors = -body + box_pole_up + box_pole_down - beam_pipe + beam_pipe_left + beam_pipe_right

        self.lower_bound = np.array([self.xmin, self.ymin, self.zmin])
        self.upper_bound = np.array([self.xmax, self.ymax, self.zmax])

        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0

        self.z_inj_beam = self.zmin + dz
    

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.
    
class Triangulation:
    def __init__(self, filename, ghost_x=20e-3, ghost_y=20e-3, ghost_z=20e-3,
                 condid=1, nz = 0):
        import meshio
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        mesh = meshio.read(filename, file_format="gmsh")
        self.points = mesh.points/1e3
        triangles2points = mesh.cells_dict['triangle']
        Ntri = np.shape(triangles2points)[0]
        triangles = self.points[triangles2points].transpose(2, 1, 0)
   
        self.xmin =  np.min(self.points[:, 0]) - ghost_x
        self.xmax = np.max(self.points[:, 0]) + ghost_x
        self.ymin = np.min(self.points[:, 1]) - ghost_y
        self.ymax = np.max(self.points[:, 1]) + ghost_y
        #self.xmin = -200e-3 - ghost_x
        #self.xmax = -self.xmin
        #self.ymin = -242e-3 / 2 - ghost_y
        #self.ymax = -self.ymin
        self.zmin = np.min(self.points[:, 2]) + ghost_z
        self.zmax = np.max(self.points[:, 2]) - ghost_z

        self.conductors = picmi.warp.Triangles(triangles, condid=condid)

        self.lower_bound = [0.97*np.min(self.points[:, 0]), 0.97*np.min(self.points[:, 1]), -0.2] #np.min(self.points[:, 2])]
        self.upper_bound = [0.97*np.max(self.points[:, 0]), 0.97*np.max(self.points[:, 1]), 0.2] #np.max(self.points[:, 2])]
        #self.lower_bound = [self.xmin, self.ymin, self.zmin]
        #self.upper_bound = [self.xmax, self.ymax, self.zmax]
        #self.lower_bound = [-0.05, -0.1, -0.2]
        #self.upper_bound = -np.array([-0.05, -0.1, -0.2])
        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0
        
        self.z_inj_beam = self.zmin + dz

    def is_outside(self, xx, yy, zz): 
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.


class CrabCavityRoundOld:
    def __init__(self, ghost_x=0, ghost_y=0, ghost_z=0, condid=1, nz = 0):
        self.ghost_x = ghost_x
        self.ghost_y = ghost_y
        self.ghost_z = ghost_z

        self.xmin =  -0.2 - ghost_x
        self.xmax = 0.2 + ghost_x
        self.ymin = -0.2 - ghost_y
        self.ymax = 0.2 + ghost_y
        self.zmin = -0.3 - ghost_z
        self.zmax = 0.3 + ghost_z

        L_int_x = 0.07
        L_int_y = 0.104
        ell = L_int_x/L_int_y
#ell = 1/ell
        h_up = h_down = 0.1

        L_out_z = 0.141/ell
        L_slope_out = 0.02/ell
        L_int_z = 0.07/ell
        L_slope_int = 0.0267/ell

        cyl_l = picmi.warp.ZCylinder(radius=0.042, zlower = 1.05*self.zmin, zupper = -0.18)
        cyl_r = picmi.warp.ZCylinder(radius=0.042, zupper = 1.05*self.zmax, zlower = 0.18)

        rminofzdata = [L_int_z+L_slope_int, L_int_z, L_int_z, L_int_z+L_slope_int]
        zmindata = [-0.284/2, -0.042,  0.042, 0.284/2]
        rmaxofzdata = [L_out_z+L_slope_out, L_out_z, L_out_z+L_slope_out]
        zmaxdata = [-0.284/2, 0, 0.284/2]
        bodyCC_rev = picmi.warp.YSrfrvEllipticInOut(ellipticity=ell, rminofydata=rminofzdata, rmaxofydata=rmaxofzdata, ymindata=zmindata, ymaxdata=zmaxdata)

        bodyCC_cyl = picmi.warp.YCylinderElliptic(ellipticity = ell, radius=L_int_z, length = 0.084)
        bodyCC= bodyCC_rev + bodyCC_cyl

        box = picmi.warp.Box(xsize=(self.xmax-self.xmin), ysize=(self.ymax-self.ymin), zsize=(self.zmax-self.zmin))
        self.conductors = box - bodyCC - cyl_l - cyl_r #cylup + cyldown #+ (box - bodyCC)

        self.lower_bound=np.array([self.xmin,self.ymin,self.zmin])
        self.upper_bound=-self.lower_bound
        if nz > 0:
            dz = (self.zmax-self.zmin)/nz
        else:
            dz = 0
        self.z_inj_beam = self.zmin + dz

    def is_outside(self, xx, yy, zz):
        return np.array(self.conductors.isinside(xx, yy, zz).isinside) == 1.
