#!/bin/bash

# Ruta al ejecutable
ejecutable="./simple"

# Inicializa el argumento en 100,000
start=100000

# Ejecuta el programa 10 veces
for i in {1..10}; do
  $ejecutable $start
  # Incrementa el argumento en 100,000 para la próxima iteración
  start=$(( start + 100000))
done
