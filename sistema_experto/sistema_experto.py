"""
Sistema Experto de RePoints: capa de reglas que valida la prediccion del
modelo de IA antes de tomar una decision final y calcula los puntos que
gana el usuario.

Estructura clasica de un sistema experto basado en reglas:
    - Base de conocimiento (clase Clase + lista de Reglas + umbrales de calidad)
    - Motor de inferencia (SistemaExperto.evaluar / evaluar_captura)
    - Explicacion de la decision (ResultadoClasificacion.explicacion)

Desde v5 el modelo (MobileNetV2/TFLite) predice una distribucion de
probabilidad sobre 3 clases -- class_names = ["papel_carton", "plastico",
"vidrio"] (orden alfabetico, ver modelo/retrain_v5_3clases.py) -- en vez de
un unico P(vidrio) binario. El sistema experto no reentrena ni toca el
modelo: solo interpreta su salida (argmax + confianza) y decide.

Puntaje (ver README / Informe de Diseno):
    plastico     -> 5 puntos
    vidrio       -> 10 puntos
    papel_carton -> 0 puntos (clase distractora, se reconoce pero no puntua)
    Si el objeto es una botella (chica o grande), el puntaje de plastico/
    vidrio se DUPLICA (ver MULTIPLICADOR_BOTELLA). No aplica a papel_carton
    (0 * cualquier cosa sigue siendo 0).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np


class Clase(Enum):
    """Clases de material que el sistema puede reconocer."""
    PLASTICO = "plastico"
    VIDRIO = "vidrio"
    PAPEL_CARTON = "papel_carton"

    def __str__(self) -> str:
        return self.value


class TamanoObjeto(Enum):
    """Tamano/forma del objeto detectado, para el bono de puntos de botella."""
    NO_ESPECIFICADO = "no_especificado"
    BOTELLA_PEQUENA = "botella_pequena"
    BOTELLA_GRANDE = "botella_grande"

    def __str__(self) -> str:
        return self.value


class Decision(Enum):
    ACEPTAR = "ACEPTAR"
    SOLICITAR_RECAPTURA = "SOLICITAR_RECAPTURA"
    DESCONOCIDO = "DESCONOCIDO"


@dataclass
class CalidadImagen:
    """Resultado de validar la captura de la camara (independiente del modelo de IA)."""
    nitidez: float          # varianza del Laplaciano (mayor = mas nitida)
    brillo: float           # promedio de gris 0-255
    es_valida: bool
    motivo: Optional[str] = None  # por que se rechazo, si es_valida=False


@dataclass
class ResultadoClasificacion:
    clase_predicha: Optional[Clase]     # None si la imagen se rechazo por calidad
    confianza: Optional[float]          # 0.0 - 1.0, confianza de clase_predicha (None si se rechazo por calidad)
    decision: Decision
    clase_final: Optional[Clase]        # None si no se acepta (recaptura/desconocido)
    puntos: float                       # 0.0 si clase_final es None o es PAPEL_CARTON
    regla_aplicada: str                 # nombre de la regla de la base de conocimiento que disparo
    explicacion: str                    # justificacion legible de la decision
    probabilidades: Optional[dict] = None  # {Clase: prob}, distribucion completa del modelo
    calidad: Optional[CalidadImagen] = None  # presente si se evaluo calidad de imagen


@dataclass
class Regla:
    """Una regla de la base de conocimiento: SI condicion(confianza) ENTONCES decision."""
    nombre: str
    condicion: Callable[[float], bool]
    decision: Decision
    plantilla_explicacion: str
    acepta_clase: bool = field(default=False)


class SistemaExperto:
    """
    Motor de inferencia por reglas para RePoints.

    Parametros configurables (para poder ajustar la sensibilidad del sistema
    sin reentrenar el modelo, tal como justifica el Informe de Diseno):

        umbral_aceptar:   confianza minima para aceptar la clase directamente.
        umbral_recaptura: confianza minima para pedir una nueva captura en vez
                           de marcar "desconocido" directamente.

    Parametros de calidad de la captura (vision/camara de la Raspberry Pi),
    mismos umbrales usados en el EDA del Informe de Analisis del Dataset:

        umbral_nitidez: varianza del Laplaciano minima para considerar la
                        imagen nitida (no borrosa). Default 100.0.
        brillo_min:     brillo medio minimo (evita imagenes muy oscuras).
                        Default 60.0.
        brillo_max:     brillo medio maximo (evita imagenes sobreexpuestas).
                        Default 200.0.

    Parametro de origen de la captura (anti-fraude, ver R0b abajo):

        resoluciones_permitidas: lista blanca de resoluciones (ancho, alto)
                        que producen las camaras/dispositivos usados para
                        capturar (en cualquier orientacion). Si la imagen
                        recibida no coincide con NINGUNA, se rechaza SIN
                        mirar nitidez/brillo ni el modelo -- una imagen
                        generada/descargada/reenviada casi nunca calza
                        exactamente con esas resoluciones fijas. Pasar None
                        explicitamente desactiva el chequeo. Si no se pasa
                        el parametro, se usan las resoluciones realmente
                        observadas en dataset_real -- (1600, 900) y
                        (1280, 960) (sesiones con la caja de luz) y
                        (1448, 1086) (camara final del proyecto usada para
                        el lote grande de materiales.rar, confirmado por el
                        usuario 2026-08-14 -- antes esta resolucion estaba
                        marcada como sospechosa por coincidir con el lote
                        fraudulento de WhatsApp del 2026-08-12; ver README).
                        Si se agrega una camara nueva, sumar su resolucion
                        aqui en vez de sacar el chequeo.

    Parametros de puntaje:

        puntos_base:    dict {Clase: puntos}. Default: plastico=5, vidrio=10,
                        papel_carton=0.
        multiplicador_botella: factor que se aplica a los puntos si el
                        objeto es una botella (chica o grande). Default 2.0
                        (duplica). Papel/carton nunca puntua, sea botella o
                        no (0 * multiplicador = 0).
    """

    _SIN_ESPECIFICAR = object()  # sentinel: distingue "no se paso el parametro" de "None explicito"

    def __init__(
        self,
        umbral_aceptar: float = 0.85,
        umbral_recaptura: float = 0.60,
        umbral_nitidez: float = 100.0,
        brillo_min: float = 60.0,
        brillo_max: float = 200.0,
        resoluciones_permitidas=_SIN_ESPECIFICAR,
        puntos_base: Optional[dict] = None,
        multiplicador_botella: float = 2.0,
    ):
        if not (0.0 <= umbral_recaptura <= umbral_aceptar <= 1.0):
            raise ValueError(
                "Se requiere 0 <= umbral_recaptura <= umbral_aceptar <= 1, "
                f"recibido umbral_recaptura={umbral_recaptura}, umbral_aceptar={umbral_aceptar}"
            )
        if brillo_min >= brillo_max:
            raise ValueError(f"brillo_min ({brillo_min}) debe ser < brillo_max ({brillo_max})")
        self.umbral_aceptar = umbral_aceptar
        self.umbral_recaptura = umbral_recaptura
        self.umbral_nitidez = umbral_nitidez
        self.brillo_min = brillo_min
        self.brillo_max = brillo_max
        if resoluciones_permitidas is SistemaExperto._SIN_ESPECIFICAR:
            self.resoluciones_permitidas = [(1600, 900), (1280, 960), (1448, 1086)]
        else:
            self.resoluciones_permitidas = resoluciones_permitidas
        self.puntos_base = puntos_base if puntos_base is not None else {
            Clase.PLASTICO: 5.0,
            Clase.VIDRIO: 10.0,
            Clase.PAPEL_CARTON: 0.0,
        }
        self.multiplicador_botella = multiplicador_botella
        self.clases = [Clase.PAPEL_CARTON, Clase.PLASTICO, Clase.VIDRIO]  # orden alfabetico, igual al modelo
        self.base_reglas = self._construir_base_reglas()

    def _construir_base_reglas(self) -> list[Regla]:
        """Base de conocimiento: reglas evaluadas en orden, gana la primera que cumpla."""
        return [
            Regla(
                nombre="R1_ACEPTAR",
                condicion=lambda c: c >= self.umbral_aceptar,
                decision=Decision.ACEPTAR,
                plantilla_explicacion=(
                    "confianza {conf:.1f}% >= umbral de aceptacion "
                    "({umbral_aceptar:.0f}%) -> se acepta la clase '{clase}' "
                    "y se clasifica mecanicamente."
                ),
                acepta_clase=True,
            ),
            Regla(
                nombre="R2_RECAPTURA",
                condicion=lambda c: c >= self.umbral_recaptura,
                decision=Decision.SOLICITAR_RECAPTURA,
                plantilla_explicacion=(
                    "confianza {conf:.1f}% esta entre el umbral de recaptura "
                    "({umbral_recaptura:.0f}%) y el de aceptacion ({umbral_aceptar:.0f}%) "
                    "-> se pide al usuario volver a acomodar el residuo para una nueva "
                    "captura, en vez de forzar la clase '{clase}'."
                ),
            ),
            Regla(
                nombre="R3_DESCONOCIDO",
                condicion=lambda c: True,  # catch-all
                decision=Decision.DESCONOCIDO,
                plantilla_explicacion=(
                    "confianza {conf:.1f}% < umbral de recaptura ({umbral_recaptura:.0f}%) "
                    "-> demasiado baja para confiar en la prediccion '{clase}'; se marca "
                    "'desconocido' en vez de forzar una clasificacion poco fiable."
                ),
            ),
        ]

    def calcular_puntos(self, clase: Optional[Clase], tamano: TamanoObjeto = TamanoObjeto.NO_ESPECIFICADO) -> float:
        """Puntos ganados por clase_final, con bono de x{multiplicador_botella} si es botella."""
        if clase is None:
            return 0.0
        base = self.puntos_base.get(clase, 0.0)
        if tamano in (TamanoObjeto.BOTELLA_PEQUENA, TamanoObjeto.BOTELLA_GRANDE):
            return base * self.multiplicador_botella
        return base

    def evaluar(
        self,
        probabilidades: Union[dict, list, np.ndarray],
        tamano: TamanoObjeto = TamanoObjeto.NO_ESPECIFICADO,
    ) -> ResultadoClasificacion:
        """
        Motor de inferencia: recibe la distribucion de probabilidad del
        modelo (softmax de 3 clases) y aplica la base de reglas en orden
        hasta encontrar la primera que dispare.

        Args:
            probabilidades: dict {Clase: prob} o secuencia alineada con
                self.clases (orden alfabetico: papel_carton, plastico,
                vidrio). Debe sumar ~1.0.
            tamano: TamanoObjeto, para el bono de puntos de botella.
        """
        if isinstance(probabilidades, dict):
            probs = {c: float(probabilidades[c]) for c in self.clases}
        else:
            probabilidades = list(probabilidades)
            if len(probabilidades) != len(self.clases):
                raise ValueError(
                    f"Se esperaban {len(self.clases)} probabilidades ({[str(c) for c in self.clases]}), "
                    f"recibido {len(probabilidades)}"
                )
            probs = {c: float(p) for c, p in zip(self.clases, probabilidades)}

        total = sum(probs.values())
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Las probabilidades deben sumar ~1.0, recibido suma={total}")
        if any(p < 0.0 or p > 1.0 for p in probs.values()):
            raise ValueError(f"Cada probabilidad debe estar en [0,1], recibido {probs}")

        clase_predicha = max(probs, key=probs.get)
        confianza = probs[clase_predicha]

        for regla in self.base_reglas:
            if regla.condicion(confianza):
                clase_final = clase_predicha if regla.acepta_clase else None
                puntos = self.calcular_puntos(clase_final, tamano)
                explicacion = regla.plantilla_explicacion.format(
                    conf=confianza * 100,
                    umbral_aceptar=self.umbral_aceptar * 100,
                    umbral_recaptura=self.umbral_recaptura * 100,
                    clase=clase_predicha,
                )
                if clase_final is not None:
                    explicacion += f" Puntos otorgados: {puntos:g} (tamano={tamano})."
                return ResultadoClasificacion(
                    clase_predicha=clase_predicha,
                    confianza=confianza,
                    decision=regla.decision,
                    clase_final=clase_final,
                    puntos=puntos,
                    regla_aplicada=regla.nombre,
                    explicacion=explicacion,
                    probabilidades=probs,
                )

        raise RuntimeError("Ninguna regla de la base de conocimiento disparo (no deberia pasar, R3 es catch-all)")

    def evaluar_calidad(self, imagen: Union[str, "np.ndarray"]) -> CalidadImagen:
        """
        R0 -- valida la captura cruda de la camara ANTES de confiarle nada al
        modelo de IA.

        R0b (primero, mas barato): la resolucion tiene que coincidir con
        alguna de las resoluciones conocidas de camara (self.resoluciones_
        permitidas). Esto NO es una medida de calidad -- es un chequeo de
        origen: fotos generadas por IA, capturas de pantalla o imagenes
        descargadas casi nunca calzan exactamente con una resolucion fija de
        camara real, y ademas suelen salir MAS nitidas que una foto real
        (sin ruido/blur de camara), asi que el filtro de nitidez de abajo no
        las detecta por si solo -- de hecho puede rechazar fotos reales
        legitimas por "borrosas" mientras deja pasar renders sospechosamente
        perfectos. Ver hallazgo documentado en el README (fotos de WhatsApp
        2026-08-12 20:41). Limitacion conocida: un ataque que imite a
        proposito una resolucion valida (ya se vio 1 caso asi) pasa este
        chequeo igual -- no reemplaza una verificacion de contenido (OCR /
        deteccion de texto ilegible) ni, idealmente, capturar directo desde
        la camara del dispositivo en vez de aceptar archivos subidos.

        R0 (nitidez/brillo): varianza del Laplaciano y brillo promedio, con
        los mismos umbrales usados en el EDA del dataset.

        Args:
            imagen: ruta a un archivo de imagen, o array BGR/gris ya cargado
                (por ejemplo con cv2.imread / la captura directa de la
                camara de la Raspberry Pi).
        """
        import cv2  # import diferido: solo hace falta si se valida calidad

        if isinstance(imagen, str):
            img = cv2.imread(imagen)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {imagen}")
        else:
            img = imagen

        if self.resoluciones_permitidas is not None:
            alto, ancho = img.shape[:2]
            actual = {ancho, alto}
            coincide = any(actual == set(res) for res in self.resoluciones_permitidas)
            if not coincide:
                gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                permitidas_str = ", ".join(f"{w}x{h}" for w, h in self.resoluciones_permitidas)
                return CalidadImagen(
                    nitidez=float(cv2.Laplacian(gray_tmp, cv2.CV_64F).var()),
                    brillo=float(np.mean(gray_tmp)),
                    es_valida=False,
                    motivo=(
                        f"resolucion {ancho}x{alto} no coincide con ninguna camara conocida "
                        f"del dispositivo ({permitidas_str} esperadas, en cualquier orientacion) "
                        "-- la imagen no parece ser una captura real del dispositivo (posible "
                        "imagen generada, reenviada o descargada)"
                    ),
                )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brillo = float(np.mean(gray))

        if nitidez < self.umbral_nitidez:
            return CalidadImagen(nitidez, brillo, es_valida=False,
                                  motivo=f"imagen borrosa (nitidez {nitidez:.1f} < {self.umbral_nitidez:.0f})")
        if brillo < self.brillo_min:
            return CalidadImagen(nitidez, brillo, es_valida=False,
                                  motivo=f"imagen muy oscura (brillo {brillo:.1f} < {self.brillo_min:.0f})")
        if brillo > self.brillo_max:
            return CalidadImagen(nitidez, brillo, es_valida=False,
                                  motivo=f"imagen sobreexpuesta (brillo {brillo:.1f} > {self.brillo_max:.0f})")
        return CalidadImagen(nitidez, brillo, es_valida=True)

    def evaluar_captura(
        self,
        probabilidades: Union[dict, list, np.ndarray],
        imagen: Optional[Union[str, "np.ndarray"]] = None,
        tamano: TamanoObjeto = TamanoObjeto.NO_ESPECIFICADO,
    ) -> ResultadoClasificacion:
        """
        Flujo completo: si se pasa `imagen`, primero corre R0 (calidad de
        camara). Si la imagen no pasa el filtro de calidad, se pide
        recaptura sin siquiera mirar la salida del modelo. Si pasa (o no se
        valida calidad porque no se paso `imagen`), se aplican las reglas de
        confianza normales (R1/R2/R3) sobre `probabilidades`.
        """
        calidad = None
        if imagen is not None:
            calidad = self.evaluar_calidad(imagen)
            if not calidad.es_valida:
                return ResultadoClasificacion(
                    clase_predicha=None,
                    confianza=None,
                    decision=Decision.SOLICITAR_RECAPTURA,
                    clase_final=None,
                    puntos=0.0,
                    regla_aplicada="R0_CALIDAD_CAMARA",
                    explicacion=(
                        f"Captura rechazada antes de clasificar: {calidad.motivo}. "
                        "Se pide recaptura sin evaluar el modelo de IA."
                    ),
                    calidad=calidad,
                )

        resultado = self.evaluar(probabilidades, tamano=tamano)
        resultado.calidad = calidad
        return resultado


# Instancia por defecto con los umbrales documentados en el Informe de Diseno
# (>=85% acepta, 60-84% recaptura, <60% desconocido), para no romper codigo
# existente que solo necesita una llamada rapida.
_sistema_por_defecto = SistemaExperto()


def clasificar_con_reglas(probabilidades: Union[dict, list, np.ndarray]) -> ResultadoClasificacion:
    """Atajo funcional sobre SistemaExperto() con los umbrales por defecto."""
    return _sistema_por_defecto.evaluar(probabilidades)


if __name__ == "__main__":
    print("=== Sistema experto por defecto (85% / 60%), modelo v5 de 3 clases ===")
    casos = [
        ({Clase.PAPEL_CARTON: 0.02, Clase.PLASTICO: 0.97, Clase.VIDRIO: 0.01}, "plastico muy seguro"),
        ({Clase.PAPEL_CARTON: 0.05, Clase.PLASTICO: 0.90, Clase.VIDRIO: 0.05}, "plastico confianza 90%"),
        ({Clase.PAPEL_CARTON: 0.15, Clase.PLASTICO: 0.70, Clase.VIDRIO: 0.15}, "plastico confianza 70%"),
        ({Clase.PAPEL_CARTON: 0.25, Clase.PLASTICO: 0.55, Clase.VIDRIO: 0.20}, "plastico confianza 55%"),
        ({Clase.PAPEL_CARTON: 0.20, Clase.PLASTICO: 0.25, Clase.VIDRIO: 0.55}, "vidrio confianza 55%"),
        ({Clase.PAPEL_CARTON: 0.10, Clase.PLASTICO: 0.20, Clase.VIDRIO: 0.70}, "vidrio confianza 70%"),
        ({Clase.PAPEL_CARTON: 0.02, Clase.PLASTICO: 0.08, Clase.VIDRIO: 0.90}, "vidrio confianza 90%"),
        ({Clase.PAPEL_CARTON: 0.01, Clase.PLASTICO: 0.01, Clase.VIDRIO: 0.98}, "vidrio muy seguro"),
        ({Clase.PAPEL_CARTON: 0.95, Clase.PLASTICO: 0.03, Clase.VIDRIO: 0.02}, "papel/carton muy seguro (0 puntos)"),
    ]
    se = SistemaExperto()
    for probs, desc in casos:
        r = se.evaluar(probs)
        print(f"{desc:32s} -> [{r.regla_aplicada}] {r.decision.value:22s} "
              f"clase_predicha={r.clase_predicha} conf={r.confianza*100:.1f}% "
              f"clase_final={r.clase_final} puntos={r.puntos:g}")
        print(f"    explicacion: {r.explicacion}")

    print("\n=== Bono de botella (duplica puntos plastico/vidrio, no afecta papel_carton) ===")
    for probs, tam, desc in [
        ({Clase.PAPEL_CARTON: 0.02, Clase.PLASTICO: 0.97, Clase.VIDRIO: 0.01}, TamanoObjeto.BOTELLA_GRANDE, "plastico, botella grande"),
        ({Clase.PAPEL_CARTON: 0.01, Clase.PLASTICO: 0.01, Clase.VIDRIO: 0.98}, TamanoObjeto.BOTELLA_PEQUENA, "vidrio, botella chica"),
        ({Clase.PAPEL_CARTON: 0.95, Clase.PLASTICO: 0.03, Clase.VIDRIO: 0.02}, TamanoObjeto.BOTELLA_GRANDE, "papel/carton, 'botella' grande (sigue en 0)"),
    ]:
        r = se.evaluar(probs, tamano=tam)
        print(f"{desc:38s} -> clase_final={r.clase_final} puntos={r.puntos:g}")

    print("\n=== Sistema experto con umbrales mas estrictos (90% / 70%), configurable sin reentrenar ===")
    se_estricto = SistemaExperto(umbral_aceptar=0.90, umbral_recaptura=0.70)
    for probs in [
        {Clase.PAPEL_CARTON: 0.05, Clase.PLASTICO: 0.08, Clase.VIDRIO: 0.87},
        {Clase.PAPEL_CARTON: 0.02, Clase.PLASTICO: 0.06, Clase.VIDRIO: 0.92},
    ]:
        r = se_estricto.evaluar(probs)
        print(f"conf={r.confianza*100:.1f}% -> [{r.regla_aplicada}] {r.decision.value} -- {r.explicacion}")

    print("\n=== R0: filtro de calidad de camara (nitidez/brillo, mismos umbrales del EDA) ===")
    import glob
    ejemplos = sorted(glob.glob(r"C:\Users\luanb\Desktop\RModel\dataset_real\plastico\*.jpeg"))[:3]
    for ruta in ejemplos:
        r = se.evaluar_captura(probabilidades={Clase.PAPEL_CARTON: 0.05, Clase.PLASTICO: 0.90, Clase.VIDRIO: 0.05}, imagen=ruta)
        print(f"{ruta.split(chr(92))[-1]}: nitidez={r.calidad.nitidez:.1f} brillo={r.calidad.brillo:.1f} "
              f"valida={r.calidad.es_valida} -> [{r.regla_aplicada}] {r.decision.value}")
