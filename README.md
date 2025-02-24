# PaliGemma2-Mix LitServe

[![Open In Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/sitammeur/studios/deploy-docci-fine-tuned-paligemma-2-vision-language-model)

[PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48) incorporates the **Gemma 2** family of language models and the **SigLIP-So400m** vision encoder, making it a significant upgrade from PaliGemma. PaliGemma 2 mix checkpoints are fine-tuned on diverse tasks and ready to use out of the box, while pt checkpoints are pre-trained and intended for further fine-tuning. These tasks include short and long captioning, optical character recognition, question answering, object detection and segmentation, and more. This project shows how to create a self-hosted, private API that deploys [PaliGemma 2 mix](https://huggingface.co/google/paligemma2-3b-mix-448) model variant with LitServe, an easy-to-use, flexible serving engine for AI models built on FastAPI.

## Project Structure

The project is structured as follows:

- `server.py`: The file containing the main code for the web server.
- `client.py`: The file containing the code for client-side requests.
- `LICENSE`: The license file for the project.
- `README.md`: The README file that contains information about the project.
- `assets`: The folder containing screenshots for working on the application.
- `images`: The folder containing images for testing purposes.
- `.env.example`: The example file for environment variables.
- `.gitignore`: The file containing the list of files and directories to be ignored by Git.

## Tech Stack

- Python (for the programming language)
- PyTorch (for the deep learning framework)
- Hugging Face Transformers Library (for the model)
- LitServe (for the serving engine)

## Getting Started

To get started with this project, follow the steps below:

1. Run the server: `python server.py`
2. Upon running the server successfully, you will see uvicorn running on port 8000.
3. Open a new terminal window.
4. Run the client: `python client.py`

Now, you can see the model's output based on the input request. The model will give you OCR results for the images in the `images` folder.

**Note**: You need a Hugging Face access token to run the application. You can get the token by signing up on the Hugging Face website and creating a new token from the settings page. After getting the token, you can set it as an environment variable `ACCESS_TOKEN` in your system by creating a `.env` file in the project's root directory. Check the `.env.example` file for reference.

## Usage

The project can be used to serve the PaliGemma 2 mix family of models using LitServe. It allows you to input an image and receive the OCR output, suggesting potential use cases in the real world, such as document scanning, text extraction, and more. The model can also be used for other tasks, including short and long captioning, question answering, object detection and segmentation, and more.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please raise an issue to discuss the changes you want to make. Once the changes are approved, you can create a pull request.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Contact

If you have any questions or suggestions about the project, feel free to contact me on my GitHub profile.

Happy coding! 🚀
