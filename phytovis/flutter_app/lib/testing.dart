import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
//import 'package:tflite/tflite.dart';
//import 'package:tflite_v2/tflite_v2.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as imgd;


class PredictionPage extends StatefulWidget {
  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  List<dynamic>? _result;
  late Interpreter loadintrptr;
  List<String> classLabels = [];
  String? clsname;

  @override
  void initState() {
    super.initState();
    loadModel();
    loadLabels();
  }

Future<void> loadLabels() async {
    try {
      String labelsFile = await rootBundle.loadString('assets/labels.txt');
      classLabels = labelsFile.split('\n');
      print("Labels loaded: ${classLabels.length}");
    } catch (e) {
      print("Error loading labels: $e");
    }
  }


//Model...

  Future<void> loadModel() async {
    try {
      String path = "assets/tfModel_v1.tflite";
      //String? loaded=await Tflite.loadModel(model:path,);
      loadintrptr = await Interpreter.fromAsset(path);

      print("\n\nModel loaded successfully::$loadintrptr");
      print("\n\n Load details complete");
    } catch (e) {
      print("\n\nError loading model: $e");
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      print("got image...");
      await _predictImage(image);
    }
  }

  Future<void> _predictImage(XFile img) async {
    try {
      //var result = await Tflite.runModelOnImage(path: img.path, numResults: 1);
      
      //preprocessImage;
      // Resize and normalize the image
      Uint8List imageBytes = await img.readAsBytes();
      var image = imgd.decodeImage(imageBytes);
      if (image != null) {
        imgd.Image resizedImage = imgd.copyResize(image, width: 256, height: 256);
        print("resized...");
        

        // Prepare the input in the shape (1, 256, 256, 3)
      List<List<List<List<double>>>> inputimg = List.generate(1, (_) =>
      List.generate(256, (_) =>
          List.generate(256, (_) => List<double>.filled(3, 0.0))));
          List<double> normalizedimg = [];
          final normalizedImage = imgd.Image(width:resizedImage.width, height:resizedImage.height);
        /*for (var pixel in resizedImage.data!) {
          int r = ((pixel as int) >> 16) & 0xFF; // Red component
          int g = ((pixel as int) >> 8) & 0xFF;  // Green component
          int b = (pixel as int) & 0xFF;  
          // Normalize RGB values to [0, 1]
          normalizedimg.add(r / 255.0); 
          normalizedimg.add(g / 255.0); 
          normalizedimg.add(b / 255.0); 
        } // If you normalized during training  */
        // Iterate through pixel data
        
        print("shapee=${resizedImage.getPixel(50,50)[0]}");
        for (int y = 0; y < resizedImage.height; y++) {
          for (int x = 0; x < resizedImage.width; x++) {
            var pixel = resizedImage.getPixel(x, y); // Get pixel value
            
            int r = pixel[0] as int;
            int g = pixel[1] as int;
            int b = pixel[2] as int; 

            //Normalize RGB values to [0, 1]
            normalizedimg.add(r / 255.0);
            normalizedimg.add(g / 255.0);
            normalizedimg.add(b / 255.0);
                  // Normalize to [0, 1]
            inputimg[0][y][x][0] = r/255.0; 
            inputimg[0][y][x][1] = g/255.0; 
            inputimg[0][y][x][2] = b/255.0;
/*            normalizedImage.setPixel(x, y, imgd.getColor(
              (normalizedR * 255).toInt(),
              (normalizedG * 255).toInt(),
              (normalizedB * 255).toInt(),
            ));*/
          }
        }
        
        print("inputimg...shape==${inputimg.shape}");
        print("normalized...shape==${normalizedimg.shape}");
        print("resizd.shape==${resizedImage}");
        print("og...shape==${image}");
      
      
      // Prepare output array
      var output = List.filled(1 * 61, 0).reshape([1, 61]); // Adjust the size based on your model's output

      // Run the model
      loadintrptr.run(inputimg, output);
      setState(() {
        _result = output;

              // Get the probabilities for the first (and only) sample
        List<double> probArray = output[0];
        print('probarray=${probArray}');
        // Find the index of the maximum probability
        int predictedClass = probArray.indexOf(probArray.reduce((a, b) => a > b ? a : b));
        clsname = classLabels[predictedClass];
        double maxValue = probArray[0];
        int maxIndex = 0;
        for (int i = 1; i < probArray.length; i++) {
          if (probArray[i] > maxValue) {
            maxValue = probArray[i];
            maxIndex = i;
          }
        }
        String? clsname2 = classLabels[maxIndex];
        print('Predicted class index: $predictedClass');
        print("Predicted label: ${clsname}"); 
        print('Predicted222${maxValue},${probArray[24]}');
        print("Predicted!!:: ${_result}");
      });

      }  //ofIf-CLause..

    } 
    catch (e) {
      print("Error predicting image: $e");
      setState(() {
        _result = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Plant Disease Predictor'),backgroundColor: Colors.green,),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.white, Colors.green], // White to Green Gradient
    	    stops: [0.75, 1.0], // White for 75%, then green
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),

        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Color.fromARGB(255, 237, 255, 237), // Light green
                    Color.fromARGB(255, 194, 253, 214), // Pale green
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                  borderRadius: BorderRadius.circular(12), // Match button corners
              ),
                child: ElevatedButton(
                  onPressed: _pickImage,
                  child: Text('Upload Image'),
                  style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent, // Set the button color trnsprnt
                  elevation: 0,
                ),
                ),
              ),
              SizedBox(height: 20),
              //Text(_result != null ? _result.toString() : 'No result'),
              Text(clsname != null ? clsname! : 'No result'),
            ],
          ),
        ),
      ),
    );
  }
}
