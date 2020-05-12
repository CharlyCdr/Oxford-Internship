import numpy as np
import networkx as nx

###############################################################################

class DynParams:
    DICTKEY_T_MAX  = 't_max'
    DICTKEY_P0 = 'p0'
    DICTKEY_W0 = 'w0'
    DICTKEY_G0 = 'g0'
    DICTKEY_T1 = 't1'
    DICTKEY_S0 = 's0'
    DICTKEY_B0 = 'b0'

    DEF_T_MAX = 100
    DEF_P0 = None
    DEF_W0 = None
    DEF_G0 = None
    DEF_T1 = None
    DEF_S0 = None
    DEF_B0 = None

    def __init__(self, t_max=None):
        self.t_max  = n_agents  if n_agents  else self.DEF_N_AGENTS


        if star:
            self.star  = star      if star[0] and star[1] else np.random.standard_normal(self.dim)
            self.star  = np.array(self.star)
        else:
            self.star  = np.random.standard_normal(self.dim)

        # TODO (WIP): make parametrizable
        self.network = nx.complete_graph(self.n_agents)

    @classmethod
    def from_dict(cls, params_dict):
        return cls(
             n_agents = params_dict[cls.DICTKEY_N_AGENTS],
             dim = params_dict[cls.DICTKEY_DIM],
             ld = params_dict[cls.DICTKEY_LAMBDA],
             star = (params_dict[cls.DICTKEY_STAR_X], params_dict[cls.DICTKEY_STAR_Y]),
             sigma = params_dict[cls.DICTKEY_SIGMA],
             timesteps = params_dict[cls.DICTKEY_TIMESTEPS],
             alpha = params_dict[cls.DICTKEY_ALPHA]
        )

###############################################################################

class AnimParams:
    DICTKEY_ZOOM_LEVEL = 'zoom_level'
    DICTKEY_ANIM_FPS   = 'anim_fps'
    DICTKEY_SAVE_PATH  = 'save_path'
    DICTKEY_SHOW       = 'show'
    DICTKEY_SHOW_FORCES= 'show_forces'
    DEF_ZOOM_LEVEL     = 1.25
    DEF_ANIM_FPS       = 10
    DEF_SHOW           = False
    DEF_SAVE_PATH      = None
    DEF_SHOW_FORCES    = False

    def __init__(self, zoom_level=None, anim_fps=None, show=None, save_path=None, show_forces=None):
        self.zoom_level = zoom_level if zoom_level else self.DEF_ZOOM_LEVEL
        self.anim_fps   = anim_fps   if anim_fps   else self.DEF_ANIM_FPS
        self.show       = show       if show       else self.DEF_SHOW
        self.save_path  = save_path  if save_path  else self.DEF_SAVE_PATH
        self.show_forces = show_forces if show_forces else self.DEF_SHOW_FORCES
    @classmethod
    def from_dict(cls, params_dict):
        return cls(
            params_dict[cls.DICTKEY_ZOOM_LEVEL],
            params_dict[cls.DICTKEY_ANIM_FPS],
            params_dict[cls.DICTKEY_SHOW],
            params_dict[cls.DICTKEY_SAVE_PATH],
        params_dict[cls.DICTKEY_SHOW_FORCES]
        )

###############################################################################
