"use client"
import React from "react";
import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowUpFromBracket, faLink } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import { NextUIProvider, Spinner } from "@nextui-org/react";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, Button, useDisclosure } from "@nextui-org/react";
import html2canvas from "html2canvas";

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState(null); // Initially null
  const [audio, setAudio] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const [backdrop, setBackdrop] = useState("blur");

  const handleOpen = () => {
    onOpen();
  };

  function onAudioChange(e) {
    const file = e.target.files[0];
    setAudio(file);
    const reader = new FileReader();

    reader.onload = () => {
      setPreviewUrl(reader.result);
    };

    if (file) {
      reader.readAsDataURL(file);
    }
  }

  const onUploadClick = async () => {
    if (!audio) {
      alert("Please select an audio file");
      return;
    }
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", audio);
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setIsLoading(false);
      handleOpen();
      console.log("File uploaded successfully:", response.data);
      if(response.data.prediction[0] === 0){
        setResult("The given audio don't have parkinson's disease");
      } else {
        setResult("The given audio has parkinson's disease");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      // Handle error
    }
  };

  
  return (
    <>
      <NextUIProvider>
        <main className="flex min-h-screen flex-col items-center sm:text-red-500 lg:text-white">
          {/* Navbar */}
          <div className="w-full px-10 py-5 border-b-2 navbar">Project Name</div>

          {/* Main Section */}
          <div className="flex flex-col gap-10 m-5 my-11 max-w-3xl">
            <h1 className="text-5xl text-wrap font-extrabold">Predict Parkinsons disease</h1>

            <h2 className="text-xl font-semibold">Upload an Audio File</h2>
            <div className="rounded-xl">
              {previewUrl && <audio controls className="w-full rounded-xl"><source src={previewUrl} type="audio/mpeg" /></audio>}
            </div>
            {/* Upload Button with an icon */}
            <div className="flex justify-end px-5">
              <input
                type="file"
                id="file"
                accept="audio/*"
                className="hidden"
                onChange={onAudioChange}
              />
              <label
                htmlFor="file"
                className="flex items-center gap-2 bg-gray-500 text-white px-4 py-2 rounded-xl cursor-pointer"
              >
                <FontAwesomeIcon icon={faArrowUpFromBracket} />
                <span>Upload Audio</span>
              </label>
            </div>

            <div className="flex justify-end px-5">
              <button
                onClick={onUploadClick}
                className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded-xl"
              >
                {isLoading ? <Spinner color="default" /> : (
                  <>
                    <FontAwesomeIcon icon={faLink} />
                    <span>Submit</span>
                  </>
                )}

              </button>
            </div>

            {/* Modal Section */}

            <Modal
              size="3xl"
              backdrop={backdrop}
              isOpen={isOpen}
              onClose={onClose}
              className="bg-gray-400 rounded-md bg-clip-padding backdrop-filter backdrop-blur-sm bg-opacity-10 border border-gray-100"
            >
              <ModalContent>
                {(onClose) => (
                  <>
                    <ModalHeader className="flex flex-col gap-1">Result</ModalHeader>
                    <ModalBody className="flex flex-row gap-5" id="modalContent">
                      <div className="w-full">
                        <h1>{result}</h1>
                      </div>
                    </ModalBody>
                    <ModalFooter>
                      <Button color="danger" variant="light" onPress={onClose}>Close</Button>
                    </ModalFooter>
                  </>
                )}
              </ModalContent>
            </Modal>
          </div>
        </main>
      </NextUIProvider>
    </>
  );
}

