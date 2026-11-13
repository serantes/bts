#!/bin/bash

# Asegurar el uso de codificación UTF-8 para procesar correctamente las tildes
export LC_ALL=en_US.UTF-8

if [ "$#" -eq 1 ]; then
    archivo="$1"
    if [ ! -f "$archivo" ]; then
        echo "Error: El archivo '$archivo' no existe." >&2
        exit 1
    fi
    # Excluye las líneas que contengan cualquier vocal acentuada (mayúscula o minúscula)
    grep -v '[áéíóúÁÉÍÓÚ]' "$archivo"
else
    # Si no se pasa archivo, lee desde la entrada estándar (stdin)
    grep -v '[áéíóúÁÉÍÓÚ]'
fi
