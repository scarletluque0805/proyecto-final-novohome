"""Genera los datos sintéticos de ventas de Novohome en Mercado Libre y los carga en SQLite.

Los datos imitan el negocio (categorías, logística Full/Colecta/Flex, precios en ARS) pero son
100% inventados: no hay clientes, órdenes ni códigos de producto reales.
Uso: python src/generar_datos.py  ->  data/novohome_ml.db
"""
import os
import sqlite3
import numpy as np
import pandas as pd

SEMILLA = 42
CATEGORIAS = {  # categoría: (precio base ARS, efecto sobre la probabilidad de devolución)
    "Pequeños electrodomésticos": (65_000, 0.30),
    "Bazar y cocina": (18_000, -0.20),
    "Grifería y plomería": (32_000, 0.90),
    "Belleza y cuidado personal": (28_000, 0.40),
    "Organización del hogar": (15_000, -0.30),
}
REGIONES = ["AMBA", "Centro", "Litoral", "Norte", "Cuyo", "Patagonia"]


def generar(n_ordenes=8000, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    # ---- productos (grano: un producto publicado) ----
    filas = []
    for i, (cat, (base, _)) in enumerate(CATEGORIAS.items()):
        for j in range(24):
            filas.append((f"NH-{i + 1}{j + 1:03d}", cat, round(base * rng.lognormal(0, 0.35), -2),
                          int(rng.integers(3, 13)),
                          float(np.clip(rng.normal(72, 12), 30, 100))))
    productos = pd.DataFrame(filas, columns=["sku", "categoria", "precio_lista", "fotos", "puntaje_ficha"])
    productos.loc[rng.random(len(productos)) < 0.08, "puntaje_ficha"] = np.nan          # ficha sin evaluar

    # ---- órdenes (grano: una orden de un producto) ----
    p = productos.sample(n_ordenes, replace=True, random_state=semilla).reset_index(drop=True)
    fecha = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 546, n_ordenes), unit="D")
    logistica = rng.choice(["Full", "Colecta", "Flex"], n_ordenes, p=[0.55, 0.30, 0.15])
    region = rng.choice(REGIONES, n_ordenes, p=[0.45, 0.18, 0.12, 0.10, 0.08, 0.07])
    lejos = np.isin(region, ["Norte", "Cuyo", "Patagonia"]).astype(int)
    dias = np.where(logistica == "Full", rng.integers(1, 4, n_ordenes),
                    np.where(logistica == "Flex", rng.integers(1, 3, n_ordenes), rng.integers(2, 8, n_ordenes))) + lejos * rng.integers(0, 4, n_ordenes)
    descuento = np.clip(rng.gamma(2, 7, n_ordenes), 0, 45).round(0)
    evento = ((fecha.month == 5) & (fecha.day <= 15)) | ((fecha.month == 11) & (fecha.day <= 10))   # Hot Sale / Cyber
    compras_previas = rng.poisson(1.2, n_ordenes)
    pregunto = rng.random(n_ordenes) < 0.25
    cuotas = rng.random(n_ordenes) < 0.5
    precio = (p.precio_lista * (1 - descuento / 100)).round(0)

    efecto_cat = p.categoria.map({k: v[1] for k, v in CATEGORIAS.items()}).values
    ficha = p.puntaje_ficha.fillna(60).values
    logit = (-3.1 + efecto_cat + 0.22 * (dias - 3) + 0.6 * (dias > 5) + 0.5 * (descuento > 30)
             - 0.035 * (ficha - 70) - 0.35 * np.log1p(compras_previas) - 0.45 * pregunto + 0.3 * evento
             + 0.9 * ((p.categoria.values == "Grifería y plomería") & (p.fotos.values < 6))
             + 0.4 * (logistica == "Colecta") * lejos)
    devolucion = rng.random(n_ordenes) < 1 / (1 + np.exp(-logit))

    ordenes = pd.DataFrame({
        "orden_id": np.arange(1, n_ordenes + 1), "fecha": fecha.strftime("%Y-%m-%d"), "sku": p.sku,
        "logistica": logistica, "region": region, "dias_entrega": dias.astype(float),
        "descuento_pct": descuento, "precio_final": precio, "cuotas_sin_interes": cuotas.astype(int),
        "compras_previas_cliente": compras_previas, "cliente_pregunto": pregunto.astype(int),
        "evento_comercial": evento.astype(int), "devolucion_o_reclamo": devolucion.astype(int),
    })
    # Columna que solo existe DESPUÉS del resultado: fuga de información (debe descartarse)
    ordenes["dias_hasta_reclamo"] = np.where(devolucion, rng.integers(1, 30, n_ordenes), np.nan)
    # ---- problemas de calidad a propósito ----
    idx = rng.permutation(n_ordenes)
    ordenes.loc[idx[:240], "dias_entrega"] = np.nan                                   # 3% sin dato de entrega
    ordenes.loc[idx[240:280], "precio_final"] *= 100                                  # 40 precios mal cargados
    ordenes.loc[idx[280:400], "region"] = ordenes.loc[idx[280:400], "region"].str.upper() + " "  # texto inconsistente
    ordenes = pd.concat([ordenes, ordenes.loc[idx[400:425]]], ignore_index=True)      # 25 órdenes duplicadas
    return productos, ordenes


def cargar_sqlite(productos, ordenes, ruta):
    """Carga ambas tablas en una transacción explícita; ante error, ROLLBACK."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    conn = sqlite3.connect(ruta, isolation_level=None)
    try:
        conn.executescript("""
            DROP TABLE IF EXISTS ordenes; DROP TABLE IF EXISTS productos;
            CREATE TABLE productos (sku TEXT PRIMARY KEY, categoria TEXT NOT NULL, precio_lista REAL CHECK (precio_lista > 0),
                                    fotos INTEGER, puntaje_ficha REAL);
            CREATE TABLE ordenes (orden_id INTEGER, fecha TEXT NOT NULL CHECK (fecha IS date(fecha)),
                                  sku TEXT NOT NULL REFERENCES productos(sku), logistica TEXT, region TEXT,
                                  dias_entrega REAL, descuento_pct REAL, precio_final REAL, cuotas_sin_interes INTEGER,
                                  compras_previas_cliente INTEGER, cliente_pregunto INTEGER, evento_comercial INTEGER,
                                  devolucion_o_reclamo INTEGER CHECK (devolucion_o_reclamo IN (0, 1)), dias_hasta_reclamo REAL);
        """)
        conn.execute("BEGIN")
        conn.executemany("INSERT INTO productos VALUES (?,?,?,?,?)", productos.astype(object).where(productos.notna(), None).values.tolist())
        conn.executemany(f"INSERT INTO ordenes VALUES ({','.join('?' * ordenes.shape[1])})",
                         ordenes.astype(object).where(ordenes.notna(), None).values.tolist())
        if conn.execute("SELECT COUNT(*) FROM ordenes").fetchone()[0] != len(ordenes):
            raise ValueError("conteo de órdenes no coincide")
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prod, ords = generar()
    cargar_sqlite(prod, ords, os.path.join(raiz, "data", "novohome_ml.db"))
    print(f"OK: {len(prod)} productos y {len(ords)} órdenes en data/novohome_ml.db")
