import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const WheatDiseaseApp());
}

class WheatDiseaseApp extends StatelessWidget {
  const WheatDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wheat Disease Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.green),
      home: const WheatDiseaseHome(),
    );
  }
}

class WheatDiseaseHome extends StatefulWidget {
  const WheatDiseaseHome({super.key});

  @override
  State<WheatDiseaseHome> createState() => _WheatDiseaseHomeState();
}

class _WheatDiseaseHomeState extends State<WheatDiseaseHome> {
  final ImagePicker _picker = ImagePicker();

  Interpreter? interpreter;
  List<String> labels = [];

  File? selectedImage;

  bool modelLoaded = false;
  bool isPredicting = false;

  String predictedLabel = "No prediction yet";
  double confidence = 0.0;

  @override
  void initState() {
    super.initState();
    loadModelAndLabels();
  }

  ///  Load both Model + Labels
  Future<void> loadModelAndLabels() async {
    try {
      debugPrint("Loading model...");

      interpreter = await Interpreter.fromAsset('assets/model.tflite');

      debugPrint(" Model loaded successfully!");

      debugPrint("Loading labels...");
      final labelData = await rootBundle.loadString("assets/labels.txt");
      labels = labelData.split("\n").where((e) => e.trim().isNotEmpty).toList();

      debugPrint(" Labels loaded: $labels");

      setState(() {
        modelLoaded = true;
      });
    } catch (e) {
      debugPrint("❌ Model/Label load error: $e");
      setState(() {
        modelLoaded = false;
      });
    }
  }

  ///  Pick image from camera
  Future<void> pickFromCamera() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() {
        selectedImage = File(image.path);
        predictedLabel = "Image selected. Ready to predict.";
        confidence = 0.0;
      });
    }
  }

  ///  Pick image from gallery
  Future<void> pickFromGallery() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        selectedImage = File(image.path);
        predictedLabel = "Image selected. Ready to predict.";
        confidence = 0.0;
      });
    }
  }

  ///  Preprocessing for MobileNetV2 (224x224 + normalize)
  List<List<List<List<double>>>> preprocessImage(File imageFile) {
    final Uint8List imageBytes = imageFile.readAsBytesSync();
    final img.Image? decoded = img.decodeImage(imageBytes);

    if (decoded == null) {
      throw Exception("Image decode failed!");
    }

    // resize
    final img.Image resized = img.copyResize(decoded, width: 224, height: 224);

    // shape: [1,224,224,3]
    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resized.getPixel(x, y);
            final r = pixel.r.toDouble() / 255.0;
            final g = pixel.g.toDouble() / 255.0;
            final b = pixel.b.toDouble() / 255.0;
            return [r, g, b];
          },
        ),
      ),
    );

    return input;
  }

  ///  Run inference
  Future<void> predict() async {
    if (!modelLoaded || interpreter == null) {
      setState(() {
        predictedLabel = " Model not loaded!";
      });
      return;
    }

    if (selectedImage == null) {
      setState(() {
        predictedLabel = " Please select an image first!";
      });
      return;
    }

    setState(() {
      isPredicting = true;
      predictedLabel = "Predicting...";
      confidence = 0.0;
    });

    try {
      final input = preprocessImage(selectedImage!);

      // output shape: [1, num_classes]
      final output = List.generate(1, (_) => List.filled(labels.length, 0.0));

      interpreter!.run(input, output);

      final scores = output[0];

      int bestIndex = 0;
      double bestScore = scores[0];

      for (int i = 1; i < scores.length; i++) {
        if (scores[i] > bestScore) {
          bestScore = scores[i];
          bestIndex = i;
        }
      }

      setState(() {
        predictedLabel = " ${labels[bestIndex]}";
        confidence = bestScore;
        isPredicting = false;
      });
    } catch (e) {
      setState(() {
        predictedLabel = "❌ Prediction error: $e";
        isPredicting = false;
      });
    }
  }

  @override
  void dispose() {
    interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Wheat Disease Detector"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // ✅ Model status
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: modelLoaded ? Colors.green.shade100 : Colors.red.shade100,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                modelLoaded ? " Model Loaded" : " Model Not Loaded",
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 16),

            //  Image Preview
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(15),
              ),
              child: selectedImage == null
                  ? const Center(child: Text("No image selected"))
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(15),
                      child: Image.file(
                        selectedImage!,
                        fit: BoxFit.cover,
                      ),
                    ),
            ),
            const SizedBox(height: 16),

            //  Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: pickFromCamera,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Camera"),
                ),
                ElevatedButton.icon(
                  onPressed: pickFromGallery,
                  icon: const Icon(Icons.image),
                  label: const Text("Gallery"),
                ),
              ],
            ),

            const SizedBox(height: 18),

            //  Predict Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: isPredicting ? null : predict,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(14),
                ),
                child: Text(isPredicting ? "Predicting..." : "Predict"),
              ),
            ),
            const SizedBox(height: 18),

            //  Output
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Prediction: $predictedLabel",
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    "Confidence: ${(confidence * 100).toStringAsFixed(2)}%",
                    style: const TextStyle(fontSize: 15),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
