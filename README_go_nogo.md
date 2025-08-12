# Script de Validación Go/No-Go - Día 2

Este script realiza una serie de pruebas para validar que la configuración del entorno de entrenamiento está lista para iniciar procesos de entrenamiento completos.

## Funcionamiento

El script ejecuta las siguientes pruebas secuenciales:

1. **Verificación de GPU** - Comprueba que hay GPUs disponibles y accesibles
2. **Benchmark del dataloader** - Prueba la velocidad y eficiencia del dataloader
3. **Prueba de vida (train_sanity.py)** - Ejecuta un entrenamiento pequeño para verificar la funcionalidad completa
4. **Evaluación de SLOs** - Analiza métricas de rendimiento contra objetivos definidos

## Ejecución

```
bash run_day2_go_nogo.sh
```

## SLOs Personalizables

Los umbrales de Service Level Objectives (SLO) se han ajustado a valores más realistas para entornos de prueba:

- **Caída de pérdida**: 0.03 (anteriormente 0.15)
  - Valor razonable para un entrenamiento de 200 pasos con un learning rate moderado
  - Puede ajustarse con `--min_loss_drop` en train_sanity.py

- **Utilización GPU**: 40% (anteriormente 85%)
  - Evita exigir saturación completa en tarjetas modernas con cargas pequeñas
  - Puede ajustarse con `--min_gpu_util` en train_sanity.py

## Logs y Monitoreo

El script genera información en consola y también guarda un registro en `go_nogo_ejecucion.log` para auditoría posterior.

Si el script utiliza monitorización de GPU, los datos se almacenan en `dmon.log` y se analizan automáticamente al final de la ejecución.

## Códigos de Salida

- **0**: Todas las pruebas pasaron correctamente (Go)
- **1**: Al menos una prueba falló (No-Go)

## Resolución de Problemas

Si el script falla en alguna etapa, revise los mensajes de error específicos en la consola y en los archivos de log para identificar el problema.
