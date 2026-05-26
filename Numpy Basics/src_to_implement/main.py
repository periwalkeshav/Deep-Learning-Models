from pattern import Circle, Checker, Spectrum
from generator import ImageGenerator

if __name__ == "__main__":
    circle = Circle(resolution=200, radius=50, position=(100, 100))
    circle.draw()
    circle.show()
    
    checker = Checker(200, 20)
    checker.draw()
    checker.show()


    spectrum = Spectrum(255)
    spectrum.draw()
    spectrum.show()
    
    img_generator = ImageGenerator(
        file_path="exercise_data",
        label_path="labels.json",
        batch_size=9,
        image_size=[32, 32, 3],
        rotation=True,
        mirroring=True,
        shuffle=True
    )

    img_generator.show()
    print("Epoch:", img_generator.current_epoch())