            # Create empty lists as fallback
            known_encodings = []
            known_names = []
            
            raise ValueError("Unable to automatically determine the structure of the data")
        except Exception as inner_e:
            print(f"Error during data extraction attempt: {inner_e}")
            exit()
            
    print(f"Successfully loaded {len(known_names)} known faces")
except Exception as e:
    print(f"Error loading known encodings: {e}")
    print("Please check the format of your pickle file. It should contain encodings and names.")
    exit()

# Load test images
image_folder = "test_images"
if not os.path.exists(image_folder):
    print(f"Error: Test image folder '{image_folder}' not found")
    exit()

test_images = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
if not test_images:
    print(f"Error: No image files found in {image_folder}")
    exit()