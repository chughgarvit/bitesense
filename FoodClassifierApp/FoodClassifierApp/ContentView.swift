import SwiftUI
import UIKit
import CoreML

struct ContentView: View {
    @State private var fileName: String = "No File Selected"
    @State private var predictedFoodType: String = ""
    @State private var isLoading: Bool = false
    @State private var showFilePicker: Bool = false

    var body: some View {
        VStack {
            Text("Food Type Classification")
                .font(.largeTitle)
                .padding()

            Text("Selected File: \(fileName)")
                .padding()

            Button(action: {
                showFilePicker = true
            }) {
                Text("Select File")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .sheet(isPresented: $showFilePicker) {
                DocumentPicker(fileName: $fileName, isLoading: $isLoading, predictedFoodType: $predictedFoodType)
            }
            
            if isLoading {
                Text("Classifying the IMU stream...")
                    .padding()
                    .font(.title)
            } else if !predictedFoodType.isEmpty {
                Text(predictedFoodType)
                    .padding()
                    .font(.title)
                    .multilineTextAlignment(.center)
            }
        }
    }
}

struct DocumentPicker: UIViewControllerRepresentable {
    @Binding var fileName: String
    @Binding var isLoading: Bool
    @Binding var predictedFoodType: String

    func makeCoordinator() -> Coordinator {
        return Coordinator(fileName: $fileName, isLoading: $isLoading, predictedFoodType: $predictedFoodType)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        // Allow selection of any file type
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.item], asCopy: true)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        @Binding var fileName: String
        @Binding var isLoading: Bool
        @Binding var predictedFoodType: String

        init(fileName: Binding<String>, isLoading: Binding<Bool>, predictedFoodType: Binding<String>) {
            _fileName = fileName
            _isLoading = isLoading
            _predictedFoodType = predictedFoodType
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let selectedFile = urls.first else { return }
            fileName = selectedFile.lastPathComponent
            isLoading = true

            // Process the file asynchronously.
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // Load file contents as string.
                    let fileContents = try String(contentsOf: selectedFile)
                    // Parse CSV to get raw IMU data.
                    let rawIMUData = self.parseCSV(fileContents)
                    
                    // Here you would normally perform filtering, thresholding, windowing,
                    // and feature extraction. For simplicity, we assume the CSV has already
                    // been preprocessed into a feature sequence with shape (seq_len, feature_dim).
                    // Adjust the following conversion as needed for your app.
                    let multiArray = try self.createMLMultiArray(from: rawIMUData)
                    
                    // Load the CoreML model (make sure "BiteSense.mlmodel" is added to your project).
                    let config = MLModelConfiguration()
                    let model = try BiteSense(configuration: config)
                    let input = BiteSenseInput(features: multiArray)
                    let prediction = try model.prediction(input: input)
                    
                    let resultString = self.formatPrediction(prediction)
                    
                    DispatchQueue.main.async {
                        self.predictedFoodType = resultString
                        self.isLoading = false
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.predictedFoodType = "Error processing file: \(error.localizedDescription)"
                        self.isLoading = false
                    }
                }
            }
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            fileName = "No File Selected"
        }
        
        // Simple CSV parser that returns an array of feature vectors.
        // Here we assume each line represents a window's preprocessed features.
        func parseCSV(_ contents: String) -> [[Double]] {
            var data: [[Double]] = []
            let lines = contents.components(separatedBy: .newlines)
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty { continue }
                let components = trimmed.split(separator: ",")
                let values = components.compactMap { Double($0) }
                // Expecting each line to have the required number of features (e.g., 90).
                if !values.isEmpty {
                    data.append(values)
                }
            }
            return data
        }
        
        // Converts a 2D array of Doubles to an MLMultiArray.
        func createMLMultiArray(from rawData: [[Double]]) throws -> MLMultiArray {
            let seqLen = rawData.count
            guard let firstRow = rawData.first else {
                throw NSError(domain: "DocumentPicker", code: -1, userInfo: [NSLocalizedDescriptionKey: "Empty data"])
            }
            let featureDim = firstRow.count
            // Create MLMultiArray with shape [seqLen, featureDim]
            let shape: [NSNumber] = [NSNumber(value: seqLen), NSNumber(value: featureDim)]
            let mlArray = try MLMultiArray(shape: shape, dataType: .double)
            for i in 0..<seqLen {
                for j in 0..<featureDim {
                    let index = i * featureDim + j
                    mlArray[index] = NSNumber(value: rawData[i][j])
                }
            }
            return mlArray
        }
        
        // Format the CoreML model output into a displayable string.
        func formatPrediction(_ prediction: BiteSenseOutput) -> String {
            // The properties below should match the output names in your CoreML model.
            // For example, assume they are numbers; adjust formatting as needed.
            let stateStr = "State: \(prediction.state)"
            let textureStr = "Texture: \(prediction.texture)"
            let nutritionalStr = "Nutritional: \(prediction.nutritional)"
            let cookingStr = "Cooking: \(prediction.cooking)"
            let foodStr = "Food Type: \(prediction.food)"
            let biteCountStr = "Bite Count: \(prediction.bite_count)"
            let durationStr = "Duration: \(prediction.duration)"
            return [stateStr, textureStr, nutritionalStr, cookingStr, foodStr, biteCountStr, durationStr].joined(separator: "\n")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
