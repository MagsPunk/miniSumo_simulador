import warnings
warnings.filterwarnings("ignore")

import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import json
import sys
import os

# ==========================================
# PALETA "DELTA ANAHUAC"
# ==========================================
BG_APP         = (12,  12,  15)
BG_PANEL       = (22,  22,  26)
NARANJA_DELTA  = (255, 85,  0)
NARANJA_HOVER  = (255, 120, 40)
BLANCO_TEXTO   = (240, 240, 240)
GRIS_TEXTO     = (150, 150, 160)
GRIS_CAJAS     = (35,  35,  40)
BORDE_CAJAS    = (60,  60,  70)

MAT_ROJO       = (160, 30,  30)
MAT_AZUL       = (25,  55,  150)
DOHYO_BASE     = (18,  18,  20)
DOHYO_LINEA    = (255, 255, 255)

ROJO_ALERTA    = (220, 40,  40)
HUD_VERDE      = (40,  220, 80)
CUCHILLA_COL   = (195, 200, 210)   

CONFIG_FILE    = "config_sumo.json"
LOGO_FILE      = "LogoDELTA.png"
MUSIC_FILE     = "arcade_theme.ogg"

# ==========================================
# FISICA DEL ENTORNO
# ==========================================
class MiniSumoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.MultiBinary(7)

        self.nombre_robot   = "Mark11_Hercules"
        self.nombre_rival   = "Rival"
        self.M              = 0.485
        self.m_llanta       = 0.022
        self.r              = 0.0145
        self.L              = 0.054
        self.R, self.Kt, self.Ke = 1.875, 0.119, 0.137
        self.V_nominal      = 9.0
        self.porcentaje_vel = 100.0
        self.rival_es_bot   = True   

        self.radio_dohyo_blanco = 0.36
        self.radio_dohyo_total  = 0.385
        self.dt = 1.0 / 60.0

        self.num_pasos        = 0
        self.traza_posiciones = []
        self.obs              = np.zeros(7, dtype=np.int8)

        self.cargar_configuracion()
        self._recalcular_J()

    def _recalcular_J(self):
        self.J            = 1e-6 + 0.5 * self.m_llanta * self.r**2 + self.M * self.r**2 / 2
        self.num_FT       = self.Kt
        self.den_FT_s     = self.J * self.R
        self.den_FT_const = self.Kt * self.Ke

    def guardar_configuracion(self):
        config = {
            "nombre_robot": self.nombre_robot,
            "nombre_rival": self.nombre_rival,
            "M": self.M, "m_llanta": self.m_llanta,
            "r": self.r, "L": self.L,
            "porcentaje_vel": self.porcentaje_vel,
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception:
            pass

    def cargar_configuracion(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    c = json.load(f)
                self.nombre_robot   = c.get("nombre_robot",   self.nombre_robot)
                self.nombre_rival   = c.get("nombre_rival",   self.nombre_rival)
                self.M              = c.get("M",              self.M)
                self.m_llanta       = c.get("m_llanta",       self.m_llanta)
                self.r              = c.get("r",              self.r)
                self.L              = c.get("L",              self.L)
                self.porcentaje_vel = c.get("porcentaje_vel", self.porcentaje_vel)
            except Exception:
                pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        d = np.random.uniform(0.25, 0.32)
        a = np.random.uniform(0, 2 * np.pi)
        self.robot_pos   = np.array([d * np.cos(a),          d * np.sin(a)])
        self.rival_pos   = np.array([d * np.cos(a + np.pi),  d * np.sin(a + np.pi)])
        self.robot_theta = np.random.uniform(0, 2 * np.pi)
        self.rival_theta = np.random.uniform(0, 2 * np.pi)
        self.num_pasos   = 0
        self.traza_posiciones = []
        self.omega_izq = self.omega_der = 0.0
        self.obs = np.zeros(7, dtype=np.int8)
        return self.obs, {}

    def step(self, action):
        fv    = np.clip(self.porcentaje_vel / 100.0, 0.0, 1.0)
        v_izq = action[0] * self.V_nominal * fv
        v_der = action[1] * self.V_nominal * fv

        tau   = self.den_FT_s / self.den_FT_const
        alpha = min(1.0, self.dt / tau)
        scale = self.num_FT / self.den_FT_const

        self.omega_izq = (1 - alpha) * self.omega_izq + alpha * v_izq * scale
        self.omega_der = (1 - alpha) * self.omega_der + alpha * v_der * scale

        v_lin = (self.omega_der + self.omega_izq) * self.r / 2.0
        v_ang = (self.omega_der - self.omega_izq) * self.r / self.L

        self.robot_theta += v_ang * self.dt
        self.robot_pos   += v_lin * np.array([np.cos(self.robot_theta),
                                               np.sin(self.robot_theta)]) * self.dt

        vel_rival = 0.35
        if self.rival_es_bot:
            vec  = self.robot_pos - self.rival_pos
            dist = np.linalg.norm(vec)
            if dist > 0:
                d = vec / dist
                self.rival_pos   += d * vel_rival * self.dt
                self.rival_theta  = np.arctan2(d[1], d[0])

        vec_col = self.rival_pos - self.robot_pos
        dist_c  = np.linalg.norm(vec_col)
        if dist_c == 0:
            dist_c = 0.001
        rc = 0.08
        if dist_c < rc:
            ov  = rc - dist_c
            sep = vec_col / dist_c
            self.robot_pos -= sep * (ov / 2)
            self.rival_pos += sep * (ov / 2)
            if v_lin > 0:
                dp = np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
                self.rival_pos += dp * v_lin * self.dt * 0.8
                self.robot_pos -= dp * v_lin * self.dt * 0.2
            if self.rival_es_bot:
                db = np.array([np.cos(self.rival_theta), np.sin(self.rival_theta)])
                self.robot_pos += db * vel_rival * self.dt * 0.8
                self.rival_pos -= db * vel_rival * self.dt * 0.2

        if np.isnan(self.robot_pos).any() or np.isnan(self.rival_pos).any():
            self.reset()
            return self.obs, 0, True, False, {}

        self.traza_posiciones.append(tuple(self.robot_pos))
        if len(self.traza_posiciones) > 40:
            self.traza_posiciones.pop(0)

        term_robot = bool(np.linalg.norm(self.robot_pos) > self.radio_dohyo_total)
        term_rival = bool(np.linalg.norm(self.rival_pos) > self.radio_dohyo_total)
        terminated = term_robot or term_rival

        if term_robot:   reward = -100.0
        elif term_rival: reward =  150.0
        else:            reward =   -0.1

        self.obs = np.zeros(7, dtype=np.int8)
        return self.obs, reward, terminated, False, \
               {'out_robot': term_robot, 'out_rival': term_rival}


# ==========================================
# INPUT BOX 
# ==========================================
class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect           = pygame.Rect(x, y, w, h)
        self.color_inactive = BORDE_CAJAS
        self.color_active   = NARANJA_DELTA
        self.color          = self.color_inactive
        self.text           = str(text)
        self.font           = pygame.font.SysFont('Consolas', 14)
        self.active         = False

    def handle_event(self, event, local_pos, scroll_y=0):
        r = self.rect.move(0, -scroll_y)
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = r.collidepoint(local_pos)
            self.color  = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, surface, scroll_y=0):
        r = self.rect.move(0, -scroll_y)
        if r.bottom < 0 or r.top > surface.get_height():
            return   
        pygame.draw.rect(surface, GRIS_CAJAS, r, border_radius=4)
        pygame.draw.rect(surface, self.color,  r, 2,  border_radius=4)
        ts = self.font.render(self.text, True, BLANCO_TEXTO)
        clip = surface.get_clip()
        surface.set_clip(r)
        surface.blit(ts, (r.x + 8, r.y + 6))
        surface.set_clip(clip)


# ==========================================
# SPLASH SCREEN
# ==========================================
class SplashScreen:
    def __init__(self, screen, virtual_surface):
        self.screen    = screen
        self.vsurf     = virtual_surface
        self.W, self.H = virtual_surface.get_size()

        self.font_title  = pygame.font.SysFont('Arial', 56, bold=True)
        self.font_sub    = pygame.font.SysFont('Arial', 28, bold=True)
        self.font_credit = pygame.font.SysFont('Consolas', 16)
        self.font_btn    = pygame.font.SysFont('Arial', 22, bold=True)

        self.logo = None
        if os.path.exists(LOGO_FILE):
            try:
                img       = pygame.image.load(LOGO_FILE).convert_alpha()
                self.logo = pygame.transform.smoothscale(img, (200, 200))
            except Exception:
                pass

        self._start_music()

        bw, bh   = 220, 56
        self.btn = pygame.Rect(self.W // 2 - bw // 2, int(self.H * 0.78), bw, bh)
        self.t   = 0.0

    def _start_music(self):
        try:
            pygame.mixer.init()
            if os.path.exists(MUSIC_FILE):
                pygame.mixer.music.load(MUSIC_FILE)
                pygame.mixer.music.set_volume(0.55)
                pygame.mixer.music.play(-1)
        except Exception:
            pass

    def _stop_music(self):
        try:
            pygame.mixer.music.fadeout(600)
        except Exception:
            pass

    def _virtual_pos(self, raw_pos):
        sw, sh = self.screen.get_size()
        return (raw_pos[0] * self.W / sw, raw_pos[1] * self.H / sh)

    def _render_to_screen(self):
        sw, sh = self.screen.get_size()
        scaled = pygame.transform.smoothscale(self.vsurf, (sw, sh))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def draw(self, hover_btn):
        self.vsurf.fill(BG_APP)
        W, H = self.W, self.H

        for i in range(0, W, 40):
            pygame.draw.line(self.vsurf, (28, 28, 34), (i, 0), (i, H))
        for j in range(0, H, 40):
            pygame.draw.line(self.vsurf, (28, 28, 34), (0, j), (W, j))

        pygame.draw.rect(self.vsurf, NARANJA_DELTA, (0, 0, W, 4))

        cy = int(H * 0.22)
        if self.logo:
            lr = self.logo.get_rect(center=(W // 2, cy))
            self.vsurf.blit(self.logo, lr)
        else:
            cx = W // 2
            r  = 80
            pts = [(cx + r * np.cos(np.pi / 6 + i * np.pi / 3),
                    cy + r * np.sin(np.pi / 6 + i * np.pi / 3)) for i in range(6)]
            pygame.draw.polygon(self.vsurf, GRIS_CAJAS,   pts)
            pygame.draw.polygon(self.vsurf, NARANJA_DELTA, pts, 3)
            lbl = self.font_sub.render("DELTA", True, NARANJA_DELTA)
            self.vsurf.blit(lbl, lbl.get_rect(center=(cx, cy)))

        title = self.font_title.render("MINI-SUMO ARENA", True, BLANCO_TEXTO)
        self.vsurf.blit(title, title.get_rect(center=(W // 2, int(H * 0.44))))

        sub = self.font_sub.render("DELTA ANAHUAC  -  Simulador Arcade", True, NARANJA_DELTA)
        self.vsurf.blit(sub, sub.get_rect(center=(W // 2, int(H * 0.54))))

        cred = self.font_credit.render("Desarrollado por MAGS", True, GRIS_TEXTO)
        self.vsurf.blit(cred, cred.get_rect(center=(W // 2, int(H * 0.62))))

        pygame.draw.line(self.vsurf, BORDE_CAJAS,
                         (W // 2 - 160, int(H * 0.68)),
                         (W // 2 + 160, int(H * 0.68)), 1)

        pulse   = 0.5 + 0.5 * np.sin(self.t * 3.0)
        btn_col = NARANJA_HOVER if hover_btn else (
            int(NARANJA_DELTA[0] * (0.85 + 0.15 * pulse)),
            int(NARANJA_DELTA[1] * (0.85 + 0.15 * pulse)), 0)

        pygame.draw.rect(self.vsurf, btn_col,      self.btn, border_radius=10)
        pygame.draw.rect(self.vsurf, BLANCO_TEXTO, self.btn, 2, border_radius=10)
        bt = self.font_btn.render(">  START", True, BLANCO_TEXTO)
        self.vsurf.blit(bt, bt.get_rect(center=self.btn.center))

        hint = self.font_credit.render(
            "Flechas / WASD  |  P = Auto-Piloto  |  B = Toggle BOT  |  R = Reiniciar",
            True, GRIS_TEXTO)
        self.vsurf.blit(hint, hint.get_rect(center=(W // 2, int(H * 0.91))))

        pygame.draw.rect(self.vsurf, NARANJA_DELTA, (0, H - 4, W, 4))
        self._render_to_screen()

    async def run(self):
        clock = pygame.time.Clock()
        while True:
            dt     = clock.tick(60) / 1000.0
            self.t += dt
            vpos   = self._virtual_pos(pygame.mouse.get_pos())
            hover  = self.btn.collidepoint(vpos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn.collidepoint(self._virtual_pos(event.pos)):
                        self._stop_music(); return True
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self._stop_music(); return True

            self.draw(hover)
            await asyncio.sleep(0)


# ==========================================
# VISUALIZADOR PRINCIPAL
# ==========================================
class SumoVisualizer:

    def __init__(self, env: MiniSumoEnv, screen, virtual_surface):
        self.env    = env
        self.screen = screen
        self.vsurf  = virtual_surface

        self._recalcular_layout()

        self.font         = pygame.font.SysFont('Arial', 14)
        self.font_bold    = pygame.font.SysFont('Arial', 14, bold=True)
        self.font_title   = pygame.font.SysFont('Arial', 17, bold=True)
        self.font_anuncio = pygame.font.SysFont('Arial', 44, bold=True)
        self.font_cd      = pygame.font.SysFont('Arial', 130, bold=True)

        self.ronda_actual    = 1
        self.victorias_robot = 0
        self.victorias_rival = 0
        self.modo            = 0       

        self.scroll_y  = 0
        self.max_scroll = 0

        self._crear_formulario()
        self.env.reset()

    def _recalcular_layout(self):
        W, H = self.vsurf.get_size()
        self.ANCHO_PANEL = max(300, min(420, int(W * 0.32)))
        self.ANCHO_VIS   = W - self.ANCHO_PANEL
        self.ESCALA      = (H * 0.88) / (2 * self.env.radio_dohyo_total)
        self.CX          = self.ANCHO_VIS // 2
        self.CY          = H // 2
        self.H           = H
        self.W           = W
        robot_px         = int(0.10 * self.ESCALA)
        self.ROBOT_PX    = max(24, min(robot_px, 80))

    def _crear_formulario(self):
        W    = self.ANCHO_PANEL
        px0  = 14
        col1 = px0                  
        iw_full = W - px0 * 2      
        iw_half = int(iw_full * 0.47)
        col2    = col1 + iw_half + int(W * 0.04)  

        Y_NOM   = 122
        Y_NRIV  = 174
        Y_M_R   = 256
        Y_L_V   = 308
        Y_BTN   = 356

        self.inputs = {}
        self.inputs['Nombre']    = InputBox(col1, Y_NOM,  iw_full, 28, self.env.nombre_robot)
        self.inputs['NombreRiv'] = InputBox(col1, Y_NRIV, iw_full, 28, self.env.nombre_rival)
        self.inputs['M']         = InputBox(col1, Y_M_R,  iw_half, 28, self.env.M)
        self.inputs['r']         = InputBox(col2, Y_M_R,  iw_half, 28, self.env.r)
        self.inputs['L']         = InputBox(col1, Y_L_V,  iw_half, 28, self.env.L)
        self.inputs['Vel']       = InputBox(col2, Y_L_V,  iw_half, 28, self.env.porcentaje_vel)

        self.btn_guardar = pygame.Rect(px0, Y_BTN, W - px0 * 2, 42)

        self._px0    = px0
        self._col1   = col1
        self._col2   = col2
        self._iw_half = iw_half

    def _world_to_pix(self, wx, wy):
        return int(wx * self.ESCALA) + self.CX, int(-wy * self.ESCALA) + self.CY

    def _aplicar_valores(self):
        try:
            self.env.nombre_robot   = self.inputs['Nombre'].text
            self.env.nombre_rival   = self.inputs['NombreRiv'].text
            self.env.M              = float(self.inputs['M'].text)
            self.env.r              = float(self.inputs['r'].text)
            self.env.L              = float(self.inputs['L'].text)
            self.env.porcentaje_vel = float(self.inputs['Vel'].text)
            self.env._recalcular_J()
            self.env.guardar_configuracion()
        except ValueError:
            pass

    def _render_to_screen(self):
        sw, sh = self.screen.get_size()
        scaled = pygame.transform.smoothscale(self.vsurf, (sw, sh))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def _draw_robot_sprite(self, surf, pos_m, theta, color_body, color_wheel,
                            name_str, is_player=False):
        px, py = self._world_to_pix(pos_m[0], pos_m[1])
        sz = self.ROBOT_PX + 10          

        chassis = pygame.Surface((sz, sz), pygame.SRCALPHA)

        margin = sz // 6
        body   = pygame.Rect(margin, margin, sz - 2 * margin, sz - 2 * margin)
        pygame.draw.rect(chassis, color_body, body, border_radius=3)
        border_col = BLANCO_TEXTO if is_player else (190, 190, 200)
        pygame.draw.rect(chassis, border_col, body, 2, border_radius=3)

        ww = max(4, sz // 6)
        wh = sz * 2 // 3
        wy = margin
        pygame.draw.rect(chassis, color_wheel, (margin - ww, wy, ww, wh), border_radius=2)
        pygame.draw.rect(chassis, color_wheel, (sz - margin, wy, ww, wh), border_radius=2)

        blade_h  = max(3, sz // 8)         
        blade_w  = sz - 2 * margin         
        blade_x  = sz - margin - blade_h   
        blade_y  = margin
        blade_rect = pygame.Rect(blade_x, blade_y, blade_h, blade_w)
        pygame.draw.rect(chassis, CUCHILLA_COL, blade_rect, border_radius=1)
        mid_bx = blade_x + blade_h // 2
        pygame.draw.line(chassis, (230, 235, 245),
                         (mid_bx, blade_y + 2),
                         (mid_bx, blade_y + blade_w - 2), 1)

        rot = pygame.transform.rotate(chassis, np.degrees(theta))
        surf.blit(rot, (px - rot.get_width() // 2, py - rot.get_height() // 2))

        tag   = self.font_bold.render(name_str, True, BLANCO_TEXTO)
        tag_x = px - tag.get_width() // 2
        tag_y = py - rot.get_height() // 2 - tag.get_height() - 3
        bg    = pygame.Rect(tag_x - 4, tag_y - 2, tag.get_width() + 8, tag.get_height() + 4)
        pygame.draw.rect(surf, BG_APP, bg, border_radius=3)
        surf.blit(tag, (tag_x, tag_y))

    def _draw_panel(self, sy):
        W, H    = self.ANCHO_PANEL, self.H
        panel   = pygame.Surface((W, H))
        panel.fill(BG_PANEL)
        pygame.draw.line(panel, BORDE_CAJAS, (0, 0), (0, H), 2)

        px0   = self._px0
        font  = self.font
        fontb = self.font_bold
        fontt = self.font_title

        def txt(text, x, y, color=BLANCO_TEXTO, f=None):
            yy = y - sy
            if -30 < yy < H + 10:
                s = (f or font).render(text, True, color)
                panel.blit(s, (x, yy))

        def sep(y):
            yy = y - sy
            if 0 < yy < H:
                pygame.draw.line(panel, BORDE_CAJAS, (px0, yy), (W - px0, yy), 1)

        y = 14
        txt("CONTROL", px0, y, NARANJA_DELTA, fontt);  y += 22
        modo_str = "MANUAL  (Flechas)" if self.modo == 0 else "AUTO-PILOTO  (P)"
        txt(f"Modo:  {modo_str}", px0, y, BLANCO_TEXTO, fontb);  y += 20
        bot_str  = "BOT AGRESIVO  ON" if self.env.rival_es_bot else "MANUAL / ESTATICO"
        bot_col  = ROJO_ALERTA if self.env.rival_es_bot else GRIS_TEXTO
        txt(f"Rival: {bot_str}", px0, y, bot_col, fontb);  y += 20
        sep(y);  y += 8

        txt("IDENTIFICACION", px0, y, NARANJA_DELTA, fontt);  y += 22
        txt("Nombre Robot:", px0, y, GRIS_TEXTO)
        y = 158   
        txt("Nombre Rival:", px0, y, GRIS_TEXTO)
        y = 210   
        sep(y);  y += 8

        txt("PARAMETROS FISICOS", px0, y, NARANJA_DELTA, fontt);  y += 22

        col2 = self._col2
        txt("Masa (kg):", px0, y, GRIS_TEXTO)
        txt("Radio (m):", col2, y, GRIS_TEXTO)
        y = 292   
        txt("Via L (m):", px0, y, GRIS_TEXTO)
        txt("Vel. (%):", col2, y, GRIS_TEXTO)
        y = 344   
        sep(y);  y += 6

        br = self.btn_guardar.move(0, -sy)
        if -60 < br.y < H + 10:
            pygame.draw.rect(panel, NARANJA_DELTA, br, border_radius=8)
            bt = fontb.render("GUARDAR  Y  APLICAR", True, BLANCO_TEXTO)
            panel.blit(bt, (br.x + (br.w - bt.get_width()) // 2, br.y + 12))

        y = self.btn_guardar.bottom + 14
        sep(y);  y += 8

        txt("CONTROLES RAPIDOS", px0, y, NARANJA_DELTA, fontt);  y += 22
        controles = [
            ("Flechas",  "Mover Robot"),
            ("W A S D",  "Mover Rival (manual)"),
            ("P",        "Activar Auto-Piloto"),
            ("B",        "Toggle BOT rival"),
            ("R",        "Reiniciar combate"),
        ]
        for tecla, desc in controles:
            ts = fontb.render(tecla, True, HUD_VERDE)
            ds = font.render(f"  {desc}", True, GRIS_TEXTO)
            yy = y - sy
            if -20 < yy < H + 10:
                panel.blit(ts, (px0, yy))
                panel.blit(ds, (px0 + ts.get_width(), yy))
            y += 20

        for box in self.inputs.values():
            box.draw(panel, sy)

        content_h = y + 20
        if content_h > H:
            self.max_scroll = content_h - H
            bar_h = max(30, int((H / content_h) * H))
            bar_y = int((sy / self.max_scroll) * (H - bar_h)) if self.max_scroll > 0 else 0
            pygame.draw.rect(panel, BORDE_CAJAS, (W - 7, bar_y, 4, bar_h), border_radius=3)
        else:
            self.max_scroll = 0

        return panel

    def draw(self, action, texto_centro=None):
        self.vsurf.fill(BG_APP)

        vis = pygame.Surface((self.ANCHO_VIS, self.H))

        pygame.draw.rect(vis, MAT_ROJO, (0,                0, self.ANCHO_VIS // 2, self.H))
        pygame.draw.rect(vis, MAT_AZUL, (self.ANCHO_VIS // 2, 0, self.ANCHO_VIS // 2, self.H))

        dsurf       = pygame.Surface((self.ANCHO_VIS, self.H), pygame.SRCALPHA)
        c           = (self.CX, self.CY)
        r_total_px  = int(self.env.radio_dohyo_total * self.ESCALA)
        pygame.draw.circle(dsurf, DOHYO_BASE, c, r_total_px)

        spacing = max(16, int(0.04 * self.ESCALA))
        for i in range(0, self.ANCHO_VIS, spacing):
            pygame.draw.line(dsurf, (36, 36, 42), (i, 0), (i, self.H), 1)
        for j in range(0, self.H, spacing):
            pygame.draw.line(dsurf, (36, 36, 42), (0, j), (self.ANCHO_VIS, j), 1)

        mask = pygame.Surface((self.ANCHO_VIS, self.H), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), c, r_total_px)
        dsurf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        border_w = max(6, int(0.012 * self.ESCALA))
        pygame.draw.circle(dsurf, DOHYO_LINEA, c, r_total_px, border_w)

        llen = int(0.04 * self.ESCALA)
        pygame.draw.line(dsurf, NARANJA_DELTA, (self.CX - llen, self.CY - 3), (self.CX + llen, self.CY - 3), 3)
        pygame.draw.line(dsurf, NARANJA_DELTA, (self.CX - llen, self.CY + 3), (self.CX + llen, self.CY + 3), 3)

        vis.blit(dsurf, (0, 0))

        if len(self.env.traza_posiciones) > 2:
            pts = [self._world_to_pix(p[0], p[1]) for p in self.env.traza_posiciones]
            for k in range(1, len(pts)):
                a = int(160 * k / len(pts))
                pygame.draw.line(vis, (255, 85, 0, a), pts[k - 1], pts[k], 1)

        if not (np.abs(self.env.rival_pos) > 1.3).any():
            self._draw_robot_sprite(vis, self.env.rival_pos, self.env.rival_theta,
                                    GRIS_CAJAS, (80, 80, 92),
                                    self.env.nombre_rival, is_player=False)

        self._draw_robot_sprite(vis, self.env.robot_pos, self.env.robot_theta,
                                NARANJA_DELTA, (25, 25, 25),
                                self.env.nombre_robot, is_player=True)

        marcador = (f"  {self.env.nombre_robot}  "
                    f"{self.victorias_robot}  vs  {self.victorias_rival}  "
                    f"{self.env.nombre_rival}  ")
        msf = self.font_bold.render(marcador, True, BLANCO_TEXTO)
        mr  = pygame.Rect(self.CX - msf.get_width() // 2 - 10, 12,
                          msf.get_width() + 20, msf.get_height() + 10)
        pygame.draw.rect(vis, BG_APP,         mr, border_radius=6)
        pygame.draw.rect(vis, NARANJA_DELTA,  mr, 2, border_radius=6)
        vis.blit(msf, (mr.x + 10, mr.y + 5))

        rnd = self.font.render(f"Ronda {self.ronda_actual}", True, GRIS_TEXTO)
        vis.blit(rnd, (self.CX - rnd.get_width() // 2, mr.bottom + 5))

        if texto_centro:
            big  = self.font_cd.render(texto_centro, True, NARANJA_DELTA)
            shad = self.font_cd.render(texto_centro, True, (0, 0, 0))
            bx   = self.CX - big.get_width() // 2
            by   = self.CY - big.get_height() // 2
            vis.blit(shad, (bx + 5, by + 5))
            vis.blit(big,  (bx, by))

        self.vsurf.blit(vis, (0, 0))

        panel = self._draw_panel(self.scroll_y)
        self.vsurf.blit(panel, (self.ANCHO_VIS, 0))

        self._render_to_screen()

    async def mostrar_cuenta_regresiva(self):
        for txt in ["3", "2", "1", "FIGHT!"]:
            t0 = pygame.time.get_ticks()
            while pygame.time.get_ticks() - t0 < 750:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit(); sys.exit()
                self.draw(np.zeros(2), texto_centro=txt)
                await asyncio.sleep(0)

    async def mostrar_anuncio(self, texto, color):
        t0 = pygame.time.get_ticks()
        while pygame.time.get_ticks() - t0 < 1600:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            self.draw(np.zeros(2))
            ann = self.font_anuncio.render(texto, True, color)
            ax  = self.CX - ann.get_width() // 2
            ay  = self.CY - ann.get_height() // 2
            bg  = pygame.Rect(ax - 24, ay - 14, ann.get_width() + 48, ann.get_height() + 28)
            pygame.draw.rect(self.vsurf, BG_APP, bg, border_radius=10)
            pygame.draw.rect(self.vsurf, color,  bg, 3, border_radius=10)
            self.vsurf.blit(ann, (ax, ay))
            self._render_to_screen()
            await asyncio.sleep(0)

    def _virtual_pos(self, raw_pos):
        sw, sh = self.screen.get_size()
        return raw_pos[0] * self.W / sw, raw_pos[1] * self.H / sh

    def _panel_local_pos(self, virtual_pos):
        return virtual_pos[0] - self.ANCHO_VIS, virtual_pos[1]

    async def run(self):
        clock   = pygame.time.Clock()
        action  = np.zeros(2)
        running = True

        await self.mostrar_cuenta_regresiva()

        while running:
            clock.tick(60)
            self.env.num_pasos += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_y = max(0, min(self.scroll_y - event.y * 35, self.max_scroll))

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
                    self.scroll_y += -35 if event.button == 4 else 35
                    self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

                if hasattr(event, 'pos'):
                    vp    = self._virtual_pos(event.pos)
                    lp    = self._panel_local_pos(vp)   

                    for box in self.inputs.values():
                        box.handle_event(event, lp, self.scroll_y)

                    if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                        br = self.btn_guardar.move(0, -self.scroll_y)
                        if br.collidepoint(lp):
                            self._aplicar_valores()
                            self.inputs['Nombre'].text    = self.env.nombre_robot
                            self.inputs['NombreRiv'].text = self.env.nombre_rival
                else:
                    for box in self.inputs.values():
                        box.handle_event(event, (0, 0), self.scroll_y)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.modo = 1 if self.modo != 1 else 0
                    if event.key == pygame.K_b:
                        self.env.rival_es_bot = not self.env.rival_es_bot
                    if event.key == pygame.K_r:
                        self.env.reset()
                        self.ronda_actual    = 1
                        self.victorias_robot = 0
                        self.victorias_rival = 0
                        action = np.zeros(2)
                        await self.mostrar_cuenta_regresiva()

            keys = pygame.key.get_pressed()

            if not self.env.rival_es_bot:
                step = 0.006
                dx = dy = 0
                if keys[pygame.K_w]: dy += step
                if keys[pygame.K_s]: dy -= step
                if keys[pygame.K_a]: dx -= step
                if keys[pygame.K_d]: dx += step
                if dx or dy:
                    self.env.rival_pos[0] += dx
                    self.env.rival_pos[1] += dy
                    self.env.rival_theta   = np.arctan2(-dy, dx)

            if self.modo == 1:
                vec     = self.env.rival_pos - self.env.robot_pos
                ang_obj = np.arctan2(vec[1], vec[0])
                diff    = np.arctan2(np.sin(ang_obj - self.env.robot_theta),
                                     np.cos(ang_obj - self.env.robot_theta))
                if abs(diff) > 0.15:
                    turn   = np.clip(diff / 0.8, -1.0, 1.0)
                    action = np.array([0.4 - turn * 0.6, 0.4 + turn * 0.6])
                else:
                    action = np.array([1.0, 1.0])
            else:
                action = np.zeros(2)
                if keys[pygame.K_UP]:    action += [ 1.0,  1.0]
                if keys[pygame.K_DOWN]:  action += [-1.0, -1.0]
                if keys[pygame.K_LEFT]:  action += [-0.5,  0.5]
                if keys[pygame.K_RIGHT]: action += [ 0.5, -0.5]

            action = np.clip(action, -1.0, 1.0)
            _, _, terminated, _, info = self.env.step(action)

            if terminated and 'out_robot' in info:
                if info['out_rival']:
                    self.victorias_robot += 1
                    await self.mostrar_anuncio(
                        f"YUKO!  {self.env.nombre_robot.upper()}", HUD_VERDE)
                elif info['out_robot']:
                    self.victorias_rival += 1
                    await self.mostrar_anuncio("OUT!  PUNTO PARA EL RIVAL", ROJO_ALERTA)

                if self.victorias_robot >= 2 or self.victorias_rival >= 2:
                    if self.victorias_robot >= 2:
                        ganador = self.env.nombre_robot.upper()
                        await self.mostrar_anuncio(f"CAMPEON: {ganador}!", HUD_VERDE)
                    else:
                        await self.mostrar_anuncio("RIVAL GANA EL MATCH", ROJO_ALERTA)
                    await asyncio.sleep(0.05)
                    running = False
                else:
                    self.ronda_actual += 1
                    self.env.reset()
                    action = np.zeros(2)
                    await self.mostrar_cuenta_regresiva()
            else:
                self.draw(action)

            await asyncio.sleep(0)

        pygame.quit()
        sys.exit()

async def main():
    pygame.init()
    try:
        info = pygame.display.Info()
        W, H = info.current_w, info.current_h
        if W < 200 or H < 200:
            W, H = 1280, 800
    except Exception:
        W, H = 1280, 800

    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("DELTA ANAHUAC - Mini-Sumo Arena")

    VIRT_W, VIRT_H  = 1280, 800
    virtual_surface = pygame.Surface((VIRT_W, VIRT_H))

    env = MiniSumoEnv()

    splash = SplashScreen(screen, virtual_surface)
    await splash.run()

    sim = SumoVisualizer(env, screen, virtual_surface)
    await sim.run()


if __name__ == "__main__":
    asyncio.run(main())