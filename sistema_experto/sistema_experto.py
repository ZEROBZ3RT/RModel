"""
Sistema Experto de RePoints: capa de reglas que valida la prediccion del
modelo de IA antes de tomar una decision final.

Estructura clasica de un sistema experto basado en reglas:
    - Base de conocimiento (clase Clase + lista de Reglas + umbrales de calidad)
    - Motor de inferencia (SistemaExperto.evaluar / evaluar_captura)
    - Explicacion de la decision (ResultadoClasificacion.explicacion)

El modelo (MobileNetV2/TFLite) predice unicamente P(vidrio) (ver notebook,
class_names = ["plastico", "vidrio"], label 1 = vidrio). El sistema experto
no reentrena ni toca el modelo: solo interpreta su salida.

Ademas de la confianza del modelo, el sistema experto puede validar la
CALIDAD de la captura de la camara (nitidez y brillo) antes de confiarle
nada al modelo -- mismos umbrales y metricas (varianza del Laplaciano,
brillo medio) que el Informe de Analisis del Dataset (seccion 4.3).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np


class Clase(Enum):
    """Clases de material que el sistema puede reconocer."""
    PLASTICO = "plastico"
    VIDRIO = "vidrio"

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
    confianza: Optional[float]          # 0.0 - 1.0, referida a clase_predicha (None si se rechazo por calidad)
    decision: Decision
    clase_final: Optional[Clase]      # None si no se acepta (recaptura/desconocido)
    regla_aplicada: str                # nombre de la regla de la base de conocimiento que disparo
    explicacion: str                   # justificacion legible de la decision
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
    """

    def __init__(
        self,
        umbral_aceptar: float = 0.85,
        umbral_recaptura: float = 0.60,
        umbral_nitidez: float = 100.0,
        brillo_min: float = 60.0,
        brillo_max: float = 200.0,
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

    def evaluar(self, prob_vidrio: float) -> ResultadoClasificacion:
        """
        Motor de inferencia: recibe P(vidrio) del modelo y aplica la base de
        reglas en orden hasta encontrar la primera que dispare.
        """
        if not 0.0 <= prob_vidrio <= 1.0:
            raise ValueError(f"prob_vidrio debe estar en [0,1], recibido {prob_vidrio}")

        if prob_vidrio >= 0.5:
            clase_predicha = Clase.VIDRIO
            confianza = prob_vidrio
        else:
            clase_predicha = Clase.PLASTICO
            confianza = 1.0 - prob_vidrio

        for regla in self.base_reglas:
            if regla.condicion(confianza):
                clase_final = clase_predicha if regla.acepta_clase else None
                explicacion = regla.plantilla_explicacion.format(
                    conf=confianza * 100,
                    umbral_aceptar=self.umbral_aceptar * 100,
                    umbral_recaptura=self.umbral_recaptura * 100,
                    clase=clase_predicha,
                )
                return ResultadoClasificacion(
                    clase_predicha=clase_predicha,
                    confianza=confianza,
                    decision=regla.decision,
                    clase_final=clase_final,
                    regla_aplicada=regla.nombre,
                    explicacion=explicacion,
                )

        raise RuntimeError("Ninguna regla de la base de conocimiento disparo (no deberia pasar, R3 es catch-all)")

    def evaluar_calidad(self, imagen: Union[str, "np.ndarray"]) -> CalidadImagen:
        """
        R0 -- valida la captura cruda de la camara ANTES de confiarle nada al
        modelo de IA: nitidez (varianza del Laplaciano) y brillo (promedio de
        gris), con los mismos umbrales usados en el EDA del dataset.

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
        prob_vidrio: float,
        imagen: Optional[Union[str, "np.ndarray"]] = None,
    ) -> ResultadoClasificacion:
        """
        Flujo completo: si se pasa `imagen`, primero corre R0 (calidad de
        camara). Si la imagen no pasa el filtro de calidad, se pide
        recaptura sin siquiera mirar la salida del modelo. Si pasa (o no se
        valida calidad porque no se paso `imagen`), se aplican las reglas de
        confianza normales (R1/R2/R3).
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
                    regla_aplicada="R0_CALIDAD_CAMARA",
                    explicacion=(
                        f"Captura rechazada antes de clasificar: {calidad.motivo}. "
                        "Se pide recaptura sin evaluar el modelo de IA."
                    ),
                    calidad=calidad,
                )

        resultado = self.evaluar(prob_vidrio)
        resultado.calidad = calidad
        return resultado


# Instancia por defecto con los umbrales documentados en el Informe de Diseno
# (>=85% acepta, 60-84% recaptura, <60% desconocido), para no romper codigo
# existente que solo necesita una llamada rapida.
_sistema_por_defecto = SistemaExperto()


def clasificar_con_reglas(prob_vidrio: float) -> ResultadoClasificacion:
    """Atajo funcional sobre SistemaExperto() con los umbrales por defecto."""
    return _sistema_por_defecto.evaluar(prob_vidrio)


if __name__ == "__main__":
    print("=== Sistema experto por defecto (85% / 60%) ===")
    casos = [
        (0.01, "plastico muy seguro"),
        (0.10, "plastico confianza 90%"),
        (0.30, "plastico confianza 70%"),
        (0.45, "plastico confianza 55%"),
        (0.55, "vidrio confianza 55%"),
        (0.70, "vidrio confianza 70%"),
        (0.90, "vidrio confianza 90%"),
        (0.99, "vidrio muy seguro"),
    ]
    se = SistemaExperto()
    for prob, desc in casos:
        r = se.evaluar(prob)
        print(f"P(vidrio)={prob:.2f} | {desc:22s} -> [{r.regla_aplicada}] {r.decision.value:22s} "
              f"clase_predicha={r.clase_predicha} conf={r.confianza*100:.1f}% "
              f"clase_final={r.clase_final}")
        print(f"    explicacion: {r.explicacion}")

    print("\n=== Sistema experto con umbrales mas estrictos (90% / 70%), configurable sin reentrenar ===")
    se_estricto = SistemaExperto(umbral_aceptar=0.90, umbral_recaptura=0.70)
    for prob in [0.87, 0.92]:
        r = se_estricto.evaluar(prob)
        print(f"P(vidrio)={prob:.2f} -> [{r.regla_aplicada}] {r.decision.value} -- {r.explicacion}")

    print("\n=== R0: filtro de calidad de camara (nitidez/brillo, mismos umbrales del EDA) ===")
    import glob
    ejemplos = sorted(glob.glob(r"C:\Users\luanb\Desktop\RModel\dataset_real\plastico\*.jpeg"))[:3]
    for ruta in ejemplos:
        r = se.evaluar_captura(prob_vidrio=0.75, imagen=ruta)
        print(f"{ruta.split(chr(92))[-1]}: nitidez={r.calidad.nitidez:.1f} brillo={r.calidad.brillo:.1f} "
              f"valida={r.calidad.es_valida} -> [{r.regla_aplicada}] {r.decision.value}")
