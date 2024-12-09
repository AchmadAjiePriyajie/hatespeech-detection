function validateFile() {
  var fileInput = document.getElementById("csv-file-input");
  var filePath = fileInput.value;
  var allowedExtensions = /(\.csv)$/i;

  if (!allowedExtensions.exec(filePath)) {
    alert("Please select a CSV file!");
    return false;
  }
  return true;
}
