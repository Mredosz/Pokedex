from google_vit_base.VitBaseTest import predict_image as predict_vit


print("Pikachu: " + predict_vit("../data/testing-images/pikachu.jpg"))
print("Gligar: " + predict_vit("../data/testing-images/gligar.jpg"))
print("Mr Fresh: " + predict_vit("../data/testing-images/mrfresh.jpg"))