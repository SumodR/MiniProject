import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
//import 'package:tflite/tflite.dart';
/*
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  List<dynamic>? _result;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try{
    String path = "assets/tfModel_v1.tflite";
    await Tflite.loadModel(model: path,);
    print("TFModelLoaded");
    }
    catch(e){
      print("Error loading model..: $e");
    }
  }




  Future<void> _predictImage(XFile img) async {
    var result = await Tflite.runModelOnImage(
      path: img.path,
      );
    setState(() {
      _result = result;
    });
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      await _predictImage(image);
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
              ElevatedButton(
                onPressed: _pickImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green, // Green Button
                ),
                child: Text('Upload Image'),
              ),
              SizedBox(height: 20),
              Text(_result.toString()),
            ],
          ),
        ),
      ),
    );
  }
}*/