"""
=============================================================================
SIMULADOR SINAPSIS
=============================================================================
Autor: [Antonio Gabucio López]
Fecha: [03 2026]
Versión: 1.0

Descripción:
(AG)Esta herramienta es un simulador fisiológico de sinapsis celulares, asemeja las variaciones de potenciales de accion en procesos sinápticos donde podemos elegir la naturaleza y el estímulo de células presinápticas y simular su efecto en células post-sinápticas.

Referencias Científicas y Teóricas:
- [Nombre del Autor 1, Año]. Título del artículo o libro. Revista o Editorial.
- Ecuación de [Nombre de la ecuación o modelo matemático utilizado].

Herramientas de Software (Open Source):
- Interfaz: [Librería gráfica, ej. Flet o CustomTkinter]
- Procesamiento: [Librería, ej. Pandas o NumPy]

Agradecimientos / Notas:
- Código estructurado y depurado con la asistencia de IA (Google Gemini).
=============================================================================
"""

import flet as ft
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def main(page: ft.Page):
    page.title = "Simulador Sinapsis - Laboratorio Interactivo, creado por Antonio Gabucio López"
    page.padding = 20
    page.theme_mode = "light"
    page.horizontal_alignment = "center"
    page.scroll = "auto" 

    dt = 0.05 

    # --- FORMA DEL POTENCIAL DE ACCIÓN (Original de 6ms para la Post-sináptica) ---
    t_spike_post = np.arange(0, 6, dt) 
    forma_espiga_post = np.zeros(len(t_spike_post))
    for j, ts in enumerate(t_spike_post):
        if ts < 1.0:
            forma_espiga_post[j] = -60.0 + 90.0 * np.sin((ts / 1.0) * (np.pi / 2))
        elif ts < 2.5:
            forma_espiga_post[j] = 30.0 - 115.0 * np.sin(((ts - 1.0) / 1.5) * (np.pi / 2))
        else:
            forma_espiga_post[j] = -85.0 + 15.0 * (1 - np.exp(-(ts - 2.5) / 1.0))
    
    longitud_espiga_post = len(forma_espiga_post)

    # --- MOTOR FISIOLÓGICO FÍSICO ---
    def simular_red(freq_A_kHz, num_A, freq_B_kHz, num_B, t_start_A, t_start_B, t_max_sim, E_rev_A, E_rev_B, tau_post, lambda_post):
        t = np.arange(0, t_max_sim, dt)
        v_reposo, v_umbral = -70.0, -60.0    
        v_pre_A, v_pre_B, v_post = np.full(len(t), v_reposo), np.full(len(t), v_reposo), np.full(len(t), v_reposo)
        
        def crear_espiga_pre(duracion):
            dur = max(duracion, dt * 3) 
            t_spk = np.arange(0, dur, dt)
            espiga = np.zeros(len(t_spk))
            for j, ts in enumerate(t_spk):
                ts_orig = ts * (6.0 / dur) 
                if ts_orig < 1.0:
                    espiga[j] = -60.0 + 90.0 * np.sin((ts_orig / 1.0) * (np.pi / 2))
                elif ts_orig < 2.5:
                    espiga[j] = 30.0 - 115.0 * np.sin(((ts_orig - 1.0) / 1.5) * (np.pi / 2))
                else:
                    espiga[j] = -85.0 + 15.0 * (1 - np.exp(-(ts_orig - 2.5) / 1.0))
            return espiga

        tiempos_disparo_A = []
        duracion_A = 6.0
        if freq_A_kHz > 0 and num_A > 0:
            intervalo_A = 1.0 / freq_A_kHz 
            duracion_A = min(6.0, intervalo_A * 0.95) 
            for n in range(int(num_A)): tiempos_disparo_A.append(t_start_A + n * intervalo_A)
        espiga_A = crear_espiga_pre(duracion_A)
                
        tiempos_disparo_B = []
        duracion_B = 6.0
        if freq_B_kHz > 0 and num_B > 0:
            intervalo_B = 1.0 / freq_B_kHz
            duracion_B = min(6.0, intervalo_B * 0.95)
            for n in range(int(num_B)): tiempos_disparo_B.append(t_start_B + n * intervalo_B)
        espiga_B = crear_espiga_pre(duracion_B)

        for ts in tiempos_disparo_A:
            idx = int(ts / dt)
            if 0 <= idx < len(t): v_pre_A[idx:min(idx + len(espiga_A), len(t))] = espiga_A[:min(idx + len(espiga_A), len(t)) - idx]

        for ts in tiempos_disparo_B:
            idx = int(ts / dt)
            if 0 <= idx < len(t): v_pre_B[idx:min(idx + len(espiga_B), len(t))] = espiga_B[:min(idx + len(espiga_B), len(t)) - idx]

        atenuacion = np.exp(-1.0 / lambda_post)
        
        tau_sinapsis = tau_post 
        g_calibracion = 0.165 
        
        retardo_sinaptico = 0.5

        g_syn_A, g_syn_B = np.zeros(len(t)), np.zeros(len(t))
        
        for ts in tiempos_disparo_A:
            t_llegada = ts + retardo_sinaptico
            mask = (t - t_llegada) > 0 
            g_syn_A[mask] += (g_calibracion * atenuacion) * ((t[mask] - t_llegada) / tau_sinapsis) * np.exp(1 - (t[mask] - t_llegada) / tau_sinapsis)
            
        for ts in tiempos_disparo_B:
            t_llegada = ts + retardo_sinaptico
            mask = (t - t_llegada) > 0 
            g_syn_B[mask] += (g_calibracion * atenuacion) * ((t[mask] - t_llegada) / tau_sinapsis) * np.exp(1 - (t[mask] - t_llegada) / tau_sinapsis)
                
        i = 1
        while i < len(t):
            # --- NUEVO: AUMENTO DE PERMEABILIDAD DEL CLORO ---
            # Si el receptor es de Cloro (-70mV), multiplicamos su conductancia x5 para simular
            # la inhibición por cortocircuito (shunting inhibition) frente a otros iones.
            peso_A = 5.0 if E_rev_A == -70.0 else 1.0
            peso_B = 5.0 if E_rev_B == -70.0 else 1.0

            corriente_ionica_A = (g_syn_A[i] * peso_A) * (E_rev_A - v_post[i-1])
            corriente_ionica_B = (g_syn_B[i] * peso_B) * (E_rev_B - v_post[i-1])
            
            # Integración fisiológica dominada por tau_post
            dv = (-(v_post[i-1] - v_reposo) + corriente_ionica_A + corriente_ionica_B) / tau_post * dt
            
            v_post[i] = v_post[i-1] + dv

            if v_post[i] >= v_umbral:
                fin_idx = min(i + longitud_espiga_post, len(t))
                v_post[i:fin_idx] = forma_espiga_post[:fin_idx - i] 
                i = fin_idx 
            else:
                i += 1

        return t, v_pre_A, v_pre_B, v_post, tiempos_disparo_A, tiempos_disparo_B

    def obtener_imagen_grafico(f_A, n_A, f_B, n_B, t_start_A, t_start_B, t_max_pre, t_max_post, E_rev_A, E_rev_B, tau, lam):
        fig = plt.figure(figsize=(10, 5.5))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4], hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0:2, 1:])
        ax4 = fig.add_subplot(gs[2, 1:], sharex=ax3) 
        
        t_max_sim = max(t_max_pre, t_max_post)
        t, v_A, v_B, v_post, t_disp_A, t_disp_B = simular_red(f_A, n_A, f_B, n_B, t_start_A, t_start_B, t_max_sim, E_rev_A, E_rev_B, tau, lam)
        
        ax1.plot(t, v_A, color='#0055ff', linewidth=1.5); ax1.axhline(y=-60, color='red', linestyle='--', alpha=0.3)
        ax1.set_ylim(-90, 50); ax1.set_xlim(0, t_max_pre); ax1.set_ylabel("mV", fontsize=8)
        ax1.set_title("Emisor A (Pre)", fontsize=10, color='#0055ff'); ax1.grid(True, linestyle='-', alpha=0.3)
        
        ax2.plot(t, v_B, color='#e63946', linewidth=1.5); ax2.axhline(y=-60, color='red', linestyle='--', alpha=0.3)
        ax2.set_ylim(-90, 50); ax2.set_xlim(0, t_max_pre); ax2.set_xlabel("ms", fontsize=8); ax2.set_ylabel("mV", fontsize=8)
        ax2.set_title("Emisor B (Pre)", fontsize=10, color='#e63946'); ax2.grid(True, linestyle='-', alpha=0.3)
        
        ax3.plot(t, v_post, color='#8a2be2', linewidth=2) 
        ax3.axhline(y=-60, color='red', linestyle='--', alpha=0.6, label='Umbral (-60mV)')
        ax3.axhline(y=-70, color='green', linestyle=':', alpha=0.4, label='Reposo (-70mV)')
        
        idx_extremo = np.argmax(np.abs(v_post - (-70.0)))
        v_extremo = v_post[idx_extremo]
        
        if abs(v_extremo - (-70.0)) > 0.05:
            ax3.axhline(y=v_extremo, color='#ff8c00', linestyle='-.', linewidth=1.5, alpha=0.9, label=f'Pico Sumación: {v_extremo:.2f} mV')
            
        lim_inf = min(-75, v_extremo - 2) if v_extremo < -70 else -75
        ax3.set_ylim(lim_inf, -55) 
        
        ax3.set_xlim(0, t_max_post) 
        ax3.set_ylabel("Voltaje (mV)", fontweight='bold')
        ax3.set_title("Receptor Post-Sináptico (Zoom Subumbral)", fontsize=12, color='#8a2be2')
        ax3.grid(True, linestyle='-', alpha=0.3); ax3.legend(loc="upper right", fontsize=8)
        
        ax4.set_ylim(0, 3)
        ax4.set_xlim(0, t_max_post)
        
        for ts in t_disp_A:
            if ts + 0.5 <= t_max_post:
                ax4.vlines(ts + 0.5, ymin=1.6, ymax=2.4, color='#0055ff', linewidth=2)
        for ts in t_disp_B:
            if ts + 0.5 <= t_max_post:
                ax4.vlines(ts + 0.5, ymin=0.6, ymax=1.4, color='#e63946', linewidth=2)
                
        ax4.set_yticks([1, 2])
        ax4.set_yticklabels(['B', 'A'], fontweight='bold', fontsize=10)
        ax4.tick_params(axis='y', length=0) 
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.set_xlabel("Tiempo (ms)", fontweight='bold')
        ax4.set_title("Llegada de neurotransmisores (Retardo sináptico: +0.5 ms)", fontsize=9, color='grey', pad=3)
        ax4.grid(True, axis='x', linestyle='-', alpha=0.3)
        plt.setp(ax3.get_xticklabels(), visible=False)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
# --- CONTROLES TÁCTILES Y RESPONSIVOS ---
    class SpinBox:
        def __init__(self, val_inicial, etiqueta, step, ancho_tf=75, min_val=0.0): # Ancho ligeramente mayor
            self.tf = ft.TextField(
                value=str(val_inicial), label=etiqueta, width=ancho_tf, dense=True, 
                text_align=ft.TextAlign.CENTER, keyboard_type=ft.KeyboardType.NUMBER
            )
            self.step = step
            self.min_val = min_val
            self.on_change = None
            
            # AUMENTAMOS el tamaño de los botones (size y padding) para que se puedan tocar con el dedo
            btn_up = ft.Container(content=ft.Text("▲", size=14, weight="bold", color="#0d0d0d"), on_click=self.up, bgcolor="#e0e0e0", padding=10, border_radius=4, ink=True)
            btn_down = ft.Container(content=ft.Text("▼", size=14, weight="bold", color="#0d0d0d"), on_click=self.down, bgcolor="#e0e0e0", padding=10, border_radius=4, ink=True)
            self.view = ft.Row([self.tf, ft.Column([btn_up, btn_down], spacing=2)], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        def up(self, e): self._change(self.step)
        def down(self, e): self._change(-self.step)
            
        def _change(self, delta):
            try: v = float(self.tf.value)
            except ValueError: v = 0.0
            new_val = max(self.min_val, v + delta)
            self.tf.value = str(int(new_val)) if new_val.is_integer() else str(round(new_val, 3))
            self.tf.update()
            if self.on_change: self.on_change()
            
        @property
        def value(self): return self.tf.value
        @value.setter
        def value(self, val): self.tf.value = str(val); self.tf.update()
        def set_on_change(self, func):
            self.on_change = func
            self.tf.on_submit = lambda e: func()
            self.tf.on_blur = lambda e: func()

    opciones_receptores = [
        ft.dropdown.Option(key="0", text="Na+/K+ (0mV)"),
        ft.dropdown.Option(key="60", text="Sodio (+60mV)"),
        ft.dropdown.Option(key="120", text="Calcio (+120mV)"),
        ft.dropdown.Option(key="-70", text="Cloro (-70mV)"),
        ft.dropdown.Option(key="-85", text="K+/Cl- (-85mV)"),
        ft.dropdown.Option(key="-90", text="Potasio (-90mV)")
    ]

    sb_start_A = SpinBox(1, "Inicio", step=1)
    sb_freq_A = SpinBox(0.20, "kHz", step=0.25) 
    sb_num_A = SpinBox(1, "Disp.", step=1)
    drop_rec_A = ft.Dropdown(value="0", width=140, dense=True, options=opciones_receptores)

    sb_start_B = SpinBox(1, "Inicio", step=1)
    sb_freq_B = SpinBox(0.20, "kHz", step=0.25) 
    sb_num_B = SpinBox(0, "Disp.", step=1)
    drop_rec_B = ft.Dropdown(value="-70", width=140, dense=True, options=opciones_receptores) 

    sb_tau = SpinBox(0.4, "τ (ms)", step=0.1, ancho_tf=75, min_val=0.1)
    sb_lambda = SpinBox(1.0, "λ", step=0.1, ancho_tf=75, min_val=0.1)

    opciones_tiempo = [ft.dropdown.Option(key=str(i), text=f"{i} ms") for i in [10, 25, 50, 100, 250, 500]]
    dropdown_t_pre = ft.Dropdown(value="25", width=120, dense=True, options=opciones_tiempo)
    dropdown_t_post = ft.Dropdown(value="25", width=120, dense=True, options=opciones_tiempo)

# Quitamos el ancho fijo estricto de la imagen para que se adapte al contenedor
    img_grafico = ft.Image(
        src=f"data:image/png;base64,{obtener_imagen_grafico(0.20, 1, 0.20, 0, 1, 1, 25, 25, 0, -70, 0.4, 1)}",
        fit="contain", # <-- Volvemos a usar el texto (string) en lugar de ft.ImageFit
        expand=True 
    )
    def val(obj):
        try: return float(obj.value) if obj.value != "" else 0.0
        except ValueError: return 0.0

    def actualizar_grafico(e=None):
        nueva_imagen = obtener_imagen_grafico(
            val(sb_freq_A), val(sb_num_A), val(sb_freq_B), val(sb_num_B), 
            val(sb_start_A), val(sb_start_B), 
            float(dropdown_t_pre.value), float(dropdown_t_post.value),
            float(drop_rec_A.value), float(drop_rec_B.value), 
            val(sb_tau) if val(sb_tau) > 0 else 0.1,
            val(sb_lambda) if val(sb_lambda) > 0 else 0.1
        )
        img_grafico.src = f"data:image/png;base64,{nueva_imagen}"
        page.update()

    for sb in [sb_start_A, sb_freq_A, sb_num_A, sb_start_B, sb_freq_B, sb_num_B, sb_tau, sb_lambda]:
        sb.set_on_change(actualizar_grafico)
    for control in [drop_rec_A, drop_rec_B, dropdown_t_pre, dropdown_t_post]:
        control.on_change = actualizar_grafico

    def reset_A(e): 
        sb_start_A.value = "1"; sb_freq_A.value = "0.20"; sb_num_A.value = "0"
        actualizar_grafico()
        
    def reset_B(e): 
        sb_start_B.value = "1"; sb_freq_B.value = "0.20"; sb_num_B.value = "0"
        actualizar_grafico()

    # --- DISEÑO RESPONSIVO (Uso de wrap=True) ---
    
    # Fila de tiempos que salta de línea si no cabe
    fila_tiempos = ft.Row([
        ft.Row([ft.Text("Escala Pre (ms):", weight="bold", size=13), dropdown_t_pre]),
        ft.Row([ft.Text("Escala Post (ms):", weight="bold", size=13), dropdown_t_post])
    ], alignment=ft.MainAxisAlignment.CENTER, wrap=True)

    panel_A_interno = ft.Container(
        content=ft.Column([
            ft.Text("Emisor A (Azul)", weight="bold", color="#0055ff"),
            ft.Row([sb_start_A.view, sb_freq_A.view, sb_num_A.view, drop_rec_A], wrap=True) # wrap aquí permite que los spinbox bajen si no caben
        ]), padding=10, border=ft.Border.all(2, "#0055ff"), border_radius=8, bgcolor="#f0f8ff"
    )
    btn_onoff_A = ft.Container(content=ft.Text("Reset", color="white", weight="bold"), bgcolor="#0055ff", padding=10, border_radius=8, ink=True, on_click=reset_A)
    fila_panel_A = ft.Row([btn_onoff_A, panel_A_interno], vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)

    panel_B_interno = ft.Container(
        content=ft.Column([
            ft.Text("Emisor B (Rojo)", weight="bold", color="#e63946"),
            ft.Row([sb_start_B.view, sb_freq_B.view, sb_num_B.view, drop_rec_B], wrap=True)
        ]), padding=10, border=ft.Border.all(2, "#e63946"), border_radius=8, bgcolor="#fff0f0"
    )
    btn_onoff_B = ft.Container(content=ft.Text("Reset", color="white", weight="bold"), bgcolor="#e63946", padding=10, border_radius=8, ink=True, on_click=reset_B)
    fila_panel_B = ft.Row([btn_onoff_B, panel_B_interno], vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True)
    
    panel_parametros = ft.Container(
        content=ft.Column([
            ft.Text("Membrana Post", weight="bold", color="#333333"),
            ft.Divider(height=1, color="grey"),
            ft.Row([ft.Text("Cte. Tiempo", size=12, weight="bold"), sb_tau.view], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([ft.Text("Cte. Espacio", size=12, weight="bold"), sb_lambda.view], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ]), padding=15, border=ft.Border.all(2, "grey"), border_radius=8, bgcolor="#fafafa"
    )

    boton_play = ft.Container(
        content=ft.Row([ft.Text("▶", size=30, color="white")], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor="#2ecc71", shape=ft.BoxShape.CIRCLE, width=70, height=70, ink=True, on_click=actualizar_grafico
    )

    # Contenedor principal de controles, salta de línea automáticamente en pantallas estrechas
    bloque_controles = ft.Row(
        [
            boton_play, 
            ft.Column([fila_panel_A, fila_panel_B], spacing=10), 
            panel_parametros
        ], 
        alignment=ft.MainAxisAlignment.CENTER, 
        vertical_alignment=ft.CrossAxisAlignment.CENTER, 
        spacing=30,
        wrap=True # <- Magia responsiva
    )

    page.add(
        ft.Text("Laboratorio: Integración Sináptica Fisiológica", size=24, weight="bold", color="blue", text_align=ft.TextAlign.CENTER),
        ft.Container(content=img_grafico, padding=5, border=ft.Border.all(1, "grey"), border_radius=10, bgcolor="white", expand=True),
        fila_tiempos, 
        ft.Container(height=5),
        bloque_controles,
        ft.Container(height=20)
    )

ft.app(target=main, view=ft.AppView.WEB_BROWSER)