
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

Future<List<String>> loadLabels(String filePath) async {
  final labelsData = await rootBundle.loadString(filePath);
  return labelsData.split('\n');
}

Future<String> classifyImage(Uint8List imageBytes) async {
  // Load the model
  final interpreter = await tfl.Interpreter.fromAsset('assets/model.tflite');

  // Load the labels
  final labels = await loadLabels('assets/labels.txt');

  // Decode image
  img.Image? image = img.decodeImage(imageBytes);

  if (image == null) {
    throw Exception("Could not decode image");
  }

  // Resize image to model input size (assuming model input size is 224x224)
  img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

  // Convert image to tensor
  var input = imageToByteList(resizedImage, 224, 224);

  // Define the output (assuming output is of size [1, 1001])
  var output = List.filled(1 * 1001, 0.0).reshape([1, 1001]);

  // Run inference
  interpreter.run(input, output);

  // Process output
  var outputList = output[0];
  double maxValue = outputList.reduce((a, b) => max(a as double, b as double));
  int maxIndex = outputList.indexOf(maxValue);

  // Get the label of the predicted class
  String predictedLabel = labels[maxIndex];

  return predictedLabel;
}

// Helper function to convert image to byte list
Uint8List imageToByteList(img.Image image, int inputSize, int outputSize) {
  var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
  var buffer = Float32List.view(convertedBytes.buffer);
  int pixelIndex = 0;

  for (var i = 0; i < inputSize; i++) {
    for (var j = 0; j < inputSize; j++) {
      var pixel = image.getPixel(j, i);
      buffer[pixelIndex++] = (img.getRed(pixel) / 255.0 - 0.5) / 0.5;
      buffer[pixelIndex++] = (img.getGreen(pixel) / 255.0 - 0.5) / 0.5;
      buffer[pixelIndex++] = (img.getBlue(pixel) / 255.0 - 0.5) / 0.5;
    }
  }
  return convertedBytes.buffer.asUint8List();
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final ImagePicker _picker = ImagePicker();
  String _predictedLabel = "";
  Uint8List? _selectedImageBytes;

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      final imageBytes = await pickedFile.readAsBytes();
      setState(() {
        _selectedImageBytes = imageBytes;
        _predictedLabel="";
      });
    }
  }

  Future<void> _classifyImage() async {
    if (_selectedImageBytes != null) {
      String label = await classifyImage(_selectedImageBytes!);
      setState(() {
        _predictedLabel = label;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color.fromARGB(255, 10, 0, 57),
        title: const Text(
          'Image Classification',
          style: TextStyle(
              fontSize: 20, color: Color.fromARGB(255, 194, 217, 255)),
        ),
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topRight,
            end: Alignment.bottomLeft,
            colors: [
              Color.fromARGB(255, 25, 4, 130),
              Color.fromARGB(255, 119, 82, 254),
            ],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              if (_selectedImageBytes != null)
                Container(
                  margin: EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.4),
                        spreadRadius: 5,
                        blurRadius: 7,
                        offset: Offset(0, 3),
                      ),
                    ],
                  ),
                  child: Image.memory(_selectedImageBytes!),
                ),
              ElevatedButton(
                onPressed: _pickImage,
                child: Text('Pick Image from Gallery'),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _classifyImage,
                child: const Text(
                  'Classify',
                  style: TextStyle(
                      fontSize: 20, color: Color.fromARGB(255, 10, 0, 57)),
                ),
              ),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(8),
                child: Text(
                  ' $_predictedLabel',
                  style: const TextStyle(
                      fontSize: 20, color: Color.fromARGB(255, 209, 199, 255)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: MyHomePage(),
  ));
}
