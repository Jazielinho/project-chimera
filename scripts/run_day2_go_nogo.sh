#!/bin/bash

echo "========== Project Chimera: DÍA 2 GO/NO-GO #1 (Prueba de Vida) =========="

# Verificar requisitos
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi no está disponible. Se requiere GPU NVIDIA."
    exit 1
fi

# Crear directorios necesarios
mkdir -p data/flickr8k/processed
mkdir -p reports

echo "\n========== Paso 1: Descarga y preparación de datos =========="
python scripts/download_prepare_data.py

if [ $? -ne 0 ]; then
    echo "ERROR: Falló la descarga y preparación de datos."
    exit 1
fi

echo "\n========== Paso 2: Benchmark del dataloader =========="
python scripts/bench_dataloader.py

if [ $? -ne 0 ]; then
    echo "ERROR: Falló el benchmark del dataloader."
    exit 1
fi

echo "\n========== Paso 3: Prueba de vida (train_sanity.py) =========="
    # Usar configuración optimizada para mejor utilización de GPU y convergencia
    python -m chimera.train.train_sanity --monitor_gpu \
    --batch_size 16 \
		--learning_rate 5e-4 \
		--num_workers 8 \
		--grad_accum_steps 8 \
		--embed_dim 256

if [ $? -ne 0 ]; then
    echo "ERROR: Falló la prueba de vida."
    exit 1
fi

echo "\n========== Evaluación de SLOs =========="

# Verificar si dmon.log existe para calcular utilización promedio de GPU
if [ -f "dmon.log" ]; then
    # Calcular utilización promedio de GPU con formato numérico correcto
    GPU_UTIL_AVG=$(awk 'NR>1 {sum+=$5; n++} END {printf "%.2f", sum/n}' dmon.log)
    echo "GPU utilización promedio: ${GPU_UTIL_AVG}%"

    # Extraer el valor numérico sin % para comparación
    GPU_UTIL_NUM=$(echo "$GPU_UTIL_AVG" | sed 's/,/./g')

    # Verificar SLO de utilización de GPU
    # Usamos el mismo umbral que pasamos a train_sanity.py (30.0%)
    if (( $(echo "$GPU_UTIL_NUM >= 30" | bc -l) )); then
        echo "✅ SLO CUMPLIDO: GPU utilización ${GPU_UTIL_AVG}% ≥ 30%"
    else
        echo "❌ SLO NO CUMPLIDO: GPU utilización ${GPU_UTIL_AVG}% < 30%"
    fi
fi

echo "\n========== Reporte final =========="
echo "Revisa los resultados completos en:"
echo " - MLflow UI: ejecuta 'mlflow ui' y abre http://localhost:5000"
echo " - Benchmark del dataloader: reports/bench_dataloader.md"
echo " - KNOWN_ISSUES.md: actualiza con los resultados y ajustes realizados"

echo "\nSi todos los SLOs se cumplen: GO para continuar con el Día 3."
echo "Si algún SLO no se cumple: actualiza KNOWN_ISSUES.md con el plan de rescate."

echo "\nNo olvides actualizar CHANGELOG.md y crear un PR con los cambios."
echo "Luego, ejecuta 'git tag v0.1-sanity-pass' y 'git push origin v0.1-sanity-pass'."
