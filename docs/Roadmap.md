# Planificare pe saptamani a proiectului

| Numar saptamana | Implementari (proiectare, cod si documentare) | Livrabile (fisiere modificate/adaugate) |
|---|---|---|
| Saptamana 4 | - Rulare model pre-antrenat pe laptop<br>- Masurare performante pentru varianta de referinta<br>- Selectare cadre din setul de date | - Roadmap.md |
| Saptamana 6 | - Export model antrenat si generare date de referinta pentru testare<br>- Structura proiect C++ (CMake) si citire date de intrare<br>- Preprocesare nor de puncte si voxelizare pe GPU<br>- Implementare encoder pilloni si generare imagine BEV pe GPU | - export_onnx.py<br>- CMakeLists.txt<br>- main.cpp<br>- pointcloud_io.cpp<br>- preprocess.cu<br>- voxelize.cu<br>- pillar_ops.cu |
| Saptamana 8 | - Accelerare module ale retelei folosind TensorRT<br>- Implementare estimare miscare proprie (ego-motion)<br>- Implementare segmentare a miscarii | - trt_engine.cpp<br>- ego_motion.cu<br>- kabsch.cpp<br>- warp.cu |
| Saptamana 10 | - Implementare clustering obiecte in miscare pe GPU<br>- Integrare module intr-un pipeline complet<br>- Verificare rezultate fata de modelul original<br>- Optimizari de precizie (FP16, INT8) | - dbscan.cu<br>- pipeline.cpp<br>- test_modules.cpp<br>- int8_calibrator.cpp |
| Saptamana 12 | - Optimizari suplimentare (reutilizare date intre cadre)<br>- Masurare latenta, throughput si consum de putere pe Jetson<br>- Calcul speedup comparativ cu implementarea de referinta<br>- Testare finala si demonstratie | - streaming_cache.cpp<br>- benchmark.cpp<br>- benchmark_results.csv<br>- README.md |