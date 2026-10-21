# Datos

La base `novohome_ml.db` (SQLite) **no se versiona**: se genera automáticamente al ejecutar el notebook
o con `python src/generar_datos.py`.

Los datos son **sintéticos**, generados con semilla fija (42). Imitan la operación de Novohome en Mercado Libre
(categorías, logística Full/Colecta/Flex, precios en ARS) pero no contienen clientes, órdenes ni códigos de
producto reales. Los SKU tienen formato genérico `NH-XNNN`.

| Tabla | Grano | Filas |
|---|---|---|
| `productos` | un producto publicado | 120 |
| `ordenes` | una orden de un producto | 8.025 (incluye 25 duplicados a propósito) |

Problemas de calidad incluidos a propósito para el EDA: nulos en `dias_entrega` y `puntaje_ficha`,
precios mal cargados (×100), región con texto inconsistente, órdenes duplicadas y una columna con fuga de
información (`dias_hasta_reclamo`).
