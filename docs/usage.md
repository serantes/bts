# BTS – Book To Speech · Manual de uso

**BTS** es un cliente de síntesis de voz local basado en el modelo neuronal
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Convierte texto,
ficheros de texto y libros EPUB en audio de alta calidad, sin conexión a
internet una vez instalado el modelo.

---

## Índice

1. [Requisitos del sistema](#1-requisitos-del-sistema)
2. [Instalación](#2-instalación)
3. [Uso rápido](#3-uso-rápido)
4. [Opciones de la línea de comandos](#4-opciones-de-la-línea-de-comandos)
5. [Modos de funcionamiento](#5-modos-de-funcionamiento)
   - 5.1 [Leer texto en voz alta](#51-leer-texto-en-voz-alta)
   - 5.2 [Leer un libro EPUB](#52-leer-un-libro-epub)
   - 5.3 [Exportar audio](#53-exportar-audio)
   - 5.4 [Análisis ortográfico y diccionario de pronunciación](#54-análisis-ortográfico-y-diccionario-de-pronunciación)
6. [Idiomas y voces](#6-idiomas-y-voces)
7. [Diccionario de pronunciación](#7-diccionario-de-pronunciación)
8. [Scripts de ayuda](#8-scripts-de-ayuda)
9. [Dispositivo y rendimiento](#9-dispositivo-y-rendimiento)
10. [Códigos de salida](#10-códigos-de-salida)
11. [Ejemplos completos](#11-ejemplos-completos)

---

## 1. Requisitos del sistema

| Requisito | Versión / notas |
|---|---|
| **Python** | 3.10, 3.11 o 3.12 (se recomienda 3.12) |
| **ffmpeg** | Requerido para reproducción y exportación MP3. Instala con `apt install ffmpeg` / `brew install ffmpeg`. |
| **espeak-ng** | Requerido para los idiomas español, francés, italiano y portugués. Instala con `apt install espeak-ng`. |
| **libhunspell** | Solo para `--analyze`. Instala con `apt install libhunspell-dev`. |
| **Diccionarios Hunspell** | Solo para `--analyze`, uno por idioma (p. ej. `apt install hunspell-es`). |
| **GPU CUDA** (opcional) | Acelera la síntesis. Requiere `--device cuda` o `--device auto`. |

La primera ejecución descarga los pesos del modelo Kokoro-82M desde
HuggingFace (~330 MB). Las ejecuciones posteriores los leen de la caché local.

---

## 2. Instalación

### Instalación automática (recomendada)

```bash
git clone https://github.com/tu-usuario/bts.git
cd bts
./scripts/bts_init.sh
```

El script crea un entorno virtual en `.venv/`, instala todas las dependencias
(incluido soporte para japonés y chino) y descarga el diccionario `unidic`
necesario para japonés.

### Instalación manual

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"           # todas las dependencias
# o solo las básicas:
pip install -e "."
# o con soporte de desarrollo:
pip install -e ".[dev]"
```

Extras disponibles:

| Extra | Contenido |
|---|---|
| `all` | Japonés + chino (misaki) |
| `ja` | Solo japonés |
| `zh` | Solo chino mandarín |
| `dev` | pytest, ruff |

Tras la instalación, los comandos `bts` y `bts-tts` estarán disponibles en el
entorno virtual. Si otro programa ya usa `bts`, usa `bts-tts` para BTS. También
puedes ejecutarlo sin activar el entorno: `.venv/bin/bts-tts`.

---

## 3. Uso rápido

```bash
# Hablar un texto directamente
bts "Hola, esto es una prueba."

# Leer un fichero de texto
bts --text novela.txt

# Leer un EPUB en español con la voz por defecto
bts --epub libro.epub

# Exportar un EPUB a MP3 (un fichero por sección)
bts --epub libro.epub --output --format mp3

# Crear un proyecto editable a partir de un EPUB o un fichero de texto
bts --epub libro.epub --project
bts --text novela.txt --project lectura

# Generar los audios definidos en un proyecto
bts-tts --build lectura
bts-tts --build lectura.bts --output audios

# Ver ayuda completa
bts --help
```

### Autocompletado en Bash

Con el entorno virtual activado, carga la integración de `argcomplete` en la
sesión actual:

```bash
source .venv/bin/activate
eval "$(register-python-argcomplete bts-tts)"
```

Para tenerlo disponible en cada terminal, añade esas dos líneas a `~/.bashrc`;
en ese caso, usa la ruta absoluta al entorno virtual. Completa `bts-tts` con
Tab; ofrece opciones, idiomas, voces y subcomandos del daemon.

### Control del daemon

Las solicitudes de síntesis enviadas al daemon se procesan en orden. Mientras
una se está procesando, `status` y los controles siguen respondiendo y las
nuevas solicitudes quedan en cola.

```bash
bts daemon start       # Inicia el daemon si está apagado
bts daemon status      # Consulta el estado aunque haya una conversión activa
bts daemon pause       # Pausa la conversión en el siguiente fragmento seguro
bts daemon play        # Reanuda una conversión pausada; si no está pausada, no hace nada
bts daemon resume      # Alias de play
bts daemon stop        # Detiene el trabajo actual; conserva las solicitudes en cola
bts daemon next        # Salta el trabajo actual y continúa con el siguiente de la cola
bts daemon list        # Muestra la tarea activa y las pendientes
bts daemon delete 3    # Borra la tarea pendiente indicada por su posición
bts daemon up 4        # Adelanta una posición la tarea pendiente
bts daemon down 2      # Retrasa una posición la tarea pendiente
bts daemon move 4 2    # Mueve la tarea 4 a la posición 2
bts daemon shutdown    # Apaga el daemon
```

La tarea activa aparece como posición `1` y no se puede borrar ni mover. Si
hay una tarea activa, `move ... 1` coloca la tarea justo detrás de ella. Los
números demasiado grandes en `move` equivalen a la última posición.

---

## 4. Opciones de la línea de comandos

```
bts [opciones] [texto]
```

### Texto de entrada

| Opción | Descripción |
|---|---|
| `texto` | Texto a sintetizar, pasado directamente. |
| `--epub FILE.epub` | Ruta a un libro EPUB. Incompatible con `--text`. |
| `--text FILE.txt` | Ruta a un fichero de texto UTF-8. Incompatible con `--epub`. |
| `--project [NAME]` | Crea un proyecto YAML `.bts` para `--epub` o `--text`. Sin `NAME`, usa el nombre de la fuente; si se da un nombre sin extensión, añade `.bts`. |
| `--build PROJECT` | Genera un archivo de audio por elemento de un proyecto `.bts`. La extensión es opcional. |
| `-` | Leer desde la entrada estándar (stdin). Se puede usar también con una tubería. |

El proyecto empieza con `Title`, `SourceFile`, `Dictionary`, `Voice`,
`OutputFormat` y `FilenamePattern`. En `Elements` se crea un elemento `source`
por sección EPUB (con su índice y título); para un fichero de texto se crea
un único elemento `source` con `id: 1` y el nombre del fichero. No se
sobrescribe un proyecto existente.

`--build` genera los elementos en orden, usando `FilenamePattern`. Por defecto
los deja en una carpeta con el nombre de la fuente; `--output DIR` permite
elegir otra. Los elementos `source` que no correspondan a una sección se
ignoran y los que tengan `action: skip` no se generan. Los campos requeridos
vacíos producen un aviso y se omite ese elemento. Los valores `default` usan
los valores predeterminados de BTS. Si existe el diccionario configurado se
usa; si falta, se avisa y se continúa sin él. Si se indican `--lang`,
`--voice`, `--format` o `--dictionary`, esos valores tienen precedencia sobre
los del proyecto y se avisa cuando reemplazan un valor explícito del archivo.

### Voz y síntesis

| Opción | Valor por defecto | Descripción |
|---|---|---|
| `--lang CÓDIGO` | `es` | Código ISO del idioma (ver [§6](#6-idiomas-y-voces)). |
| `--voice VOZ` | Primera voz del idioma | ID de la voz neuronal. |
| `--speed N` | `1.0` | Multiplicador de velocidad. `0.5` = muy lento, `2.0` = muy rápido. |
| `--max-pause SEGUNDOS` | `0.5` | Duración máxima de las pausas de texto. Varias marcas seguidas producen una sola pausa. `0` desactiva estas pausas. |
| `--device cpu\|cuda\|auto` | `auto` | Dónde ejecutar el modelo. Usa la GPU si está disponible y tiene memoria libre; si no, usa la CPU. |
| `--display` | — | Muestra el texto en pantalla resaltando las palabras mientras se hablan. |

### Salida de audio

| Opción | Descripción |
|---|---|
| `--output [DIR\|FILE]` | Guarda el audio en lugar de reproducirlo. Con `--epub`: directorio (uno por sección). Sin valor: directorio con el nombre del libro. Para texto plano: nombre del fichero de salida. |
| `--format flac\|mp3` | Formato del audio exportado. Si se omite, se deduce de la extensión o se usa FLAC. |
| `--mp3-quality 0-9` | Calidad VBR del MP3 (0 = mejor/más grande, 9 = menor calidad/más compacto). Por defecto: `4`. |

### Diccionario de pronunciación

| Opción | Descripción |
|---|---|
| `--dictionary FILE` | Fichero de pronunciación (ver [§7](#7-diccionario-de-pronunciación)). Con `--analyze`, es el nombre del fichero a crear. |
| `--dictionary-ignore-case` | Aplica el diccionario sin distinguir mayúsculas. |

### Análisis ortográfico

| Opción | Descripción |
|---|---|
| `--analyze` | Genera un fichero `.dict` con las palabras desconocidas del texto. |
| `--ipa [LANG]` | Con `--analyze`, añade las fonemas IPA de Kokoro para cada palabra. Si se indica `LANG`, usa ese idioma para los fonemas (p. ej. nombres ingleses en un libro español: `--lang es --ipa us`). |

### Secciones (solo EPUB)

| Opción | Descripción |
|---|---|
| `--section-list` | Lista todas las secciones del EPUB y termina. |
| `--section-start SECCIÓN` | Empieza a leer o exportar desde el índice o subcadena de título indicado. |

### Descubrimiento e información

| Opción | Descripción |
|---|---|
| `--list-languages` | Muestra todos los códigos de idioma disponibles. |
| `--list-voices [LANG]` | Muestra las voces disponibles. Con `LANG`, filtra por idioma. |
| `--test LANG` | Prueba todas las voces de un idioma con texto de demostración. |
| `--version` | Muestra la versión instalada. |
| `-h`, `--help` | Muestra la ayuda completa. |

---

## 5. Modos de funcionamiento

### 5.1 Leer texto en voz alta

BTS puede recibir texto de tres fuentes:

**Texto en la línea de comandos:**
```bash
bts "Buenos días, ¿cómo estás?"
bts --lang us "Good morning, how are you?"
bts --lang fr --voice ff_siwis "Bonjour le monde."
```

La raya `—` no se pronuncia: por defecto inserta medio segundo de silencio.
Los marcadores y sus duraciones se configuran en `PAUSE_MARKERS` de
`src/bts/constants.py`, para poder añadir otros textos o palabras más adelante.
`--max-pause` limita la duración máxima; marcas consecutivas, como `——` o
`— —`, se agrupan en una sola pausa. Se puede usar `--max-pause 0` para
desactivar estas pausas.
El carácter invisible U+2060 se elimina del texto; se pueden añadir más
cadenas a `IGNORED_STRINGS` en `src/bts/constants.py`.

**Fichero de texto:**
```bash
bts --text articulo.txt
bts --text articulo.txt --lang us --speed 1.3
```

**Entrada estándar (stdin):**
```bash
# Mediante pipe
echo "Hola mundo" | bts
cat capitulo.txt | bts --speed 0.9

# Con '-' para leer stdin explícitamente
bts --lang us -

# Desde un fichero redirigido
bts - < informe.txt
```

> **Nota:** cuando no se pasan argumentos, BTS espera **hasta 2 segundos**
> que haya datos en stdin antes de mostrar la ayuda. Si se ejecuta en un
> cron o en segundo plano sin stdin, no se bloqueará.

### 5.2 Leer un libro EPUB

```bash
# Leer todo el libro desde el principio
bts --epub Quijote.epub

# Leer con voz y velocidad personalizadas
bts --epub Quijote.epub --voice ef_dora --speed 1.1

# Ver las secciones disponibles antes de leer
bts --epub Quijote.epub --section-list

# Comenzar desde la sección 5
bts --epub Quijote.epub --section-start 5

# Comenzar desde una sección buscando por título
bts --epub Quijote.epub --section-start "capítulo tres"
```

El título de cada sección se anuncia antes del cuerpo, a una velocidad
ligeramente inferior a la del texto para que suene como una locución natural.

### 5.3 Exportar audio

BTS puede exportar el audio a ficheros FLAC o MP3 en lugar de reproducirlo.
El audio exportado tiene la misma calidad que el reproducido en tiempo real.

**Exportar un EPUB (un fichero por sección):**
```bash
# FLAC, carpeta con el nombre del libro
bts --epub Quijote.epub --output

# MP3, en una carpeta específica
bts --epub Quijote.epub --output ~/Audios/Quijote --format mp3

# MP3 de alta calidad (VBR 0)
bts --epub Quijote.epub --output --format mp3 --mp3-quality 0

# Reanudar una exportación interrumpida (las secciones ya exportadas se saltan)
bts --epub Quijote.epub --output

# Exportar solo desde la sección 10
bts --epub Quijote.epub --output --section-start 10
```

Los ficheros se nombran automáticamente `01_Título.flac`, `02_Título.flac`, etc.
Si un fichero ya existe al reanudar, se omite sin regenerarlo.

**Exportar texto plano:**
```bash
# La extensión determina el formato
bts --text articulo.txt --output articulo.flac
bts --text articulo.txt --output articulo.mp3

# Especificando formato explícitamente
bts --text articulo.txt --output articulo.flac --format flac
```

### 5.4 Análisis ortográfico y diccionario de pronunciación

El análisis ortográfico detecta palabras que Hunspell no conoce (nombres
propios, siglas, extranjerismos, etc.) y genera un fichero `.dict` listo
para usarse como diccionario de pronunciación.

**Flujo de trabajo típico:**

```bash
# Paso 1: analizar el libro y generar el diccionario con fonemas IPA
bts --epub MiLibro.epub --analyze --ipa

# Se genera: MiLibro.dict
# Contenido de ejemplo:
#   Tolkien|Tolkien
#   McCulley|[mˈɪkʌli]
#   Zorg|[zˈɔɹɡ]

# Paso 2: editar el fichero a mano, corrigiendo los fonemas incorrectos

# Paso 3: leer el libro usando el diccionario
bts --epub MiLibro.epub --dictionary MiLibro.dict
```

**Opciones del análisis:**
```bash
# Solo obtener la lista de palabras (sin fonemas)
bts --epub libro.epub --analyze

# Con fonemas del mismo idioma que --lang
bts --epub libro.epub --lang es --analyze --ipa

# Con fonemas en otro idioma (nombres ingleses en texto español)
bts --epub libro.epub --lang es --analyze --ipa us

# Guardar el diccionario con un nombre distinto
bts --epub libro.epub --analyze --ipa --dictionary nombres.dict

# Analizar un fichero de texto
bts --text articulo.txt --lang us --analyze --ipa
```

> **Requisitos:** `libhunspell` y el diccionario Hunspell del idioma de
> `--lang` deben estar instalados. Los directorios de búsqueda son los
> habituales del sistema (`/usr/share/hunspell`, etc.) y el contenido de
> la variable de entorno `DICPATH`.

---

## 6. Idiomas y voces

```bash
# Ver todos los idiomas disponibles
bts --list-languages

# Ver todas las voces
bts --list-voices

# Ver las voces de un idioma concreto
bts --list-voices es
bts --list-voices us

# Probar todas las voces de un idioma
bts --test es
bts --test us
```

### Códigos de idioma

| Código | Idioma | Notas |
|---|---|---|
| `es` | Español | Voz por defecto |
| `en` | Inglés (británico por defecto) | Alias de `uk` |
| `uk` | Inglés británico | |
| `us` | Inglés americano | |
| `fr` | Francés | Requiere espeak-ng |
| `it` | Italiano | Requiere espeak-ng |
| `pt` | Portugués | Requiere espeak-ng |
| `ja` | Japonés (日本語) | Requiere extra `ja` + `python -m unidic download` |
| `zh` | Chino mandarín (中文) | Requiere extra `zh` |

### Voces disponibles

| Idioma | Voces |
|---|---|
| **Español** | `ef_dora`, `em_alex`, `em_santa` |
| **Inglés británico** | `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`, `bm_george`, `bm_lewis`, `bm_daniel`, `bm_fable` |
| **Inglés americano** | `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| **Francés** | `ff_siwis` |
| **Italiano** | `if_sara`, `im_nicola` |
| **Japonés** | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo` |
| **Chino mandarín** | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` |
| **Portugués** | `pf_dora`, `pm_alex`, `pm_santa` |

---

## 7. Diccionario de pronunciación

Un diccionario de pronunciación es un fichero de texto UTF-8 con una entrada
por línea. BTS lo aplica antes de sintetizar cada fragmento de texto.

### Formato

```
# Líneas que empiezan con '#' son comentarios y se ignoran.

# Sustitución simple (palabra por palabra)
Sr|señor
Dr|doctor
EE UU|Estados Unidos

# Sustitución fonética IPA (entre corchetes, dentro de paréntesis)
Tolkien|[tˈɒlkiːn]
McCulley|[mˈɪkʌli]
```

Cuando el reemplazo va entre corchetes — `[palabra](/fonemas/)` — BTS
inyecta directamente los fonemas en el motor de Kokoro, garantizando una
pronunciación exacta. Cuando es texto plano, el motor aplica su propio
procesado G2P sobre el texto de reemplazo.

### Reglas de aplicación

- El separador entre la palabra original y el reemplazo es `|`.
- La coincidencia respeta mayúsculas por defecto. Usa `--dictionary-ignore-case` para ignorarlas.
- Los caracteres de puntuación adyacentes a la palabra no impiden la coincidencia.
- La sustitución es de izquierda a derecha; las entradas no se encadenan.

### Generar un diccionario con `--analyze --ipa`

La forma más rápida de crear un diccionario es analizar el texto fuente:

```bash
bts --epub libro.epub --analyze --ipa
```

Esto genera `libro.dict` con una entrada `palabra|[palabra](/fonemas/)` por
cada palabra que Hunspell no reconoce. Los fonemas son los que Kokoro usaría
por defecto, listos para revisar y corregir a mano.

---

## 8. Scripts de ayuda

La carpeta `scripts/` contiene scripts de Bash preconfigurados para los casos
de uso más comunes. Todos buscan el ejecutable `bts` en `.venv/bin/bts` y
aceptan argumentos adicionales que se pasan directamente a BTS.

### `bts_init.sh`
Crea el entorno virtual, instala todas las dependencias y descarga el
diccionario unidic para japonés.
```bash
./scripts/bts_init.sh
```

### `bts_esf` — Exportar EPUB en español a FLAC
```bash
# Uso: bts_esf LIBRO.epub [opciones extra de bts]
./scripts/bts_esf MiLibro.epub
./scripts/bts_esf MiLibro.epub --section-start 5
```
Exporta el libro a una carpeta con su nombre. Si existe un fichero
`MiLibro.dict` en el mismo directorio, lo usa como diccionario automáticamente.

### `bts_esm` — Exportar EPUB en español a MP3
Igual que `bts_esf` pero con formato MP3.
```bash
./scripts/bts_esm MiLibro.epub
```

### `bts_tts` — Acceso directo a BTS
Wrapper fino que llama a `.venv/bin/bts` con todos los argumentos que se le pasen.
```bash
./scripts/bts_tts --epub libro.epub --lang us --speed 1.2
```

---

## 9. Dispositivo y rendimiento

BTS puede ejecutar el modelo Kokoro en CPU o GPU (CUDA).

```bash
# CPU (por defecto, funciona en cualquier máquina)
bts --device cpu "Texto de prueba."

# GPU (requiere PyTorch con soporte CUDA)
bts --device cuda "Texto de prueba."

# Automático: usa GPU si tiene ≥1 GB libre, si no CPU
bts --device auto "Texto de prueba."
```

### Tiempos orientativos

| Hardware | ~palabras/segundo |
|---|---|
| CPU moderna (8 núcleos) | 20–40 |
| GPU NVIDIA (RTX 3060 o mejor) | 150–400 |

La carga del modelo es **diferida**: `--help`, `--version`, `--list-languages`,
`--list-voices` y `--section-list` responden instantáneamente sin cargar los
pesos del modelo.

### Memoria de GPU

El modelo Kokoro-82M requiere aproximadamente 830 MB de VRAM. BTS necesita al
menos 1 GB libre para aceptar el modo `cuda`. Con `--device auto`, si la GPU
no tiene esa memoria disponible, cae automáticamente a CPU.

Cuando la GPU se queda sin memoria durante la síntesis, BTS imprime una
advertencia y termina con código de salida 0 (para facilitar reintentos desde
scripts).

---

## 10. Códigos de salida

| Código | Significado |
|---|---|
| `0` | Éxito, o GPU sin memoria (para permitir reintentos). |
| `1` | Error (fichero no encontrado, idioma no soportado, etc.). |
| `2` | Error de uso (argumento inválido). |

---

## 11. Ejemplos completos

### Leer una novela en español

```bash
# Ver secciones disponibles
bts --epub Quijote.epub --section-list

# Escuchar desde el capítulo 3, con la voz ef_dora a velocidad 1.1
bts --epub Quijote.epub --section-start "Capítulo III" --voice ef_dora --speed 1.1
```

### Exportar una novela a MP3 de alta calidad con diccionario

```bash
# Generar diccionario con fonemas en inglés para los nombres propios
bts --epub Shogun.epub --lang es --analyze --ipa us

# Editar Shogun.dict a mano para corregir pronunciaciones

# Exportar con el diccionario corregido
bts --epub Shogun.epub --output --format mp3 --mp3-quality 0 \
    --dictionary Shogun.dict --device auto
```

### Escuchar noticias desde stdin

```bash
curl -s https://ejemplo.com/noticias.txt | bts --lang es --speed 1.2
```

### Probar una voz antes de usarla

```bash
# Probar todas las voces americanas
bts --test us

# Probar una voz específica
bts --lang us --voice af_nova "This is a test of the nova voice."
```

### Exportar un artículo a FLAC

```bash
bts --text articulo.txt --output articulo.flac
# o equivalente, con formato explícito:
bts --text articulo.txt --output articulo --format flac
```

### Integración en un script

```bash
#!/usr/bin/env bash
# Lee un resumen del tiempo cada mañana
curl -s https://api.tiempo.example/es | \
    .venv/bin/bts --lang es --speed 1.0 --device auto -
```

---

## Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar los tests
pytest

# Verificar estilo de código
ruff check src tests

# Ver la cobertura del código fuente
pytest --tb=short -q
```

Los tests no requieren GPU, red ni ffmpeg: todas las dependencias pesadas
están simuladas con mocks en `tests/conftest.py`.
