import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

Future<List<String>> loadLabels(String filePath) async {
  final labelsData = await rootBundle.loadString(filePath);
  return labelsData.split('\n');
}

Future<String> classifyImage(String imagePath) async {
  // Load the model
  final interpreter = await tfl.Interpreter.fromAsset('assets/model.tflite');

  // Load the labels
  final labels = await loadLabels('assets/labels.txt');

  // Load the image
  final ByteData imageData = await rootBundle.load(imagePath);
  final Uint8List bytes = imageData.buffer.asUint8List();

  // Decode image
  img.Image? image = img.decodeImage(bytes);

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
  int maxIndex = outputList.indexOf(outputList.reduce(max));

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

void main() async {
  // Example usage
  String predictedLabel = await classifyImage('assets/sample.jpg');
  print('Predicted label: $predictedLabel');
}
