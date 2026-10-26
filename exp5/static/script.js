// ============================================================
// VAE STUDIO - FRONTEND JAVASCRIPT
// ============================================================


let selectedFile = null;

let imageCount = 9;


// ============================================================
// FILE INPUT
// ============================================================

const fileInput =
    document.getElementById(
        "fileInput"
    );


const uploadBox =
    document.getElementById(
        "uploadBox"
    );


fileInput.addEventListener(
    "change",
    function () {

        if (
            this.files &&
            this.files.length > 0
        ) {

            selectedFile =
                this.files[0];

            showSelectedImage(
                selectedFile
            );
        }

    }
);


// ============================================================
// SHOW SELECTED IMAGE
// ============================================================

function showSelectedImage(file) {

    const reader =
        new FileReader();

    reader.onload = function (event) {

        const img =
            document.getElementById(
                "inputImage"
            );

        const placeholder =
            document.getElementById(
                "inputPlaceholder"
            );

        img.src =
            event.target.result;

        img.style.display =
            "block";

        placeholder.style.display =
            "none";
    };


    reader.readAsDataURL(file);
}


// ============================================================
// DRAG & DROP
// ============================================================

uploadBox.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadBox.style.borderColor =
            "#a8ff4d";

    }
);


uploadBox.addEventListener(
    "dragleave",
    function () {

        uploadBox.style.borderColor =
            "";

    }
);


uploadBox.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadBox.style.borderColor =
            "";

        const files =
            event.dataTransfer.files;

        if (
            files.length > 0
        ) {

            selectedFile =
                files[0];

            showSelectedImage(
                selectedFile
            );
        }

    }
);


// ============================================================
// RECONSTRUCT
// ============================================================

document
    .getElementById(
        "reconstructBtn"
    )
    .addEventListener(
        "click",
        async function () {

            if (!selectedFile) {

                alert(
                    "Please upload an MNIST digit first."
                );

                return;
            }


            const button = this;

            button.disabled =
                true;

            button.innerHTML =
                "Encoding...";


            const formData =
                new FormData();

            formData.append(
                "image",
                selectedFile
            );


            try {

                const response =
                    await fetch(
                        "/reconstruct",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                if (!data.success) {

                    throw new Error(
                        data.error
                    );
                }


                // ------------------------------------------------
                // Reconstruction
                // ------------------------------------------------

                const reconstructedImage =
                    document.getElementById(
                        "reconstructedImage"
                    );

                const reconstructionPlaceholder =
                    document.getElementById(
                        "reconstructionPlaceholder"
                    );


                reconstructedImage.src =
                    "data:image/png;base64," +
                    data.reconstruction;

                reconstructedImage.style.display =
                    "block";

                reconstructionPlaceholder.style.display =
                    "none";


                // ------------------------------------------------
                // Latent coordinates
                // ------------------------------------------------

                document.getElementById(
                    "z1"
                ).textContent =
                    data.z1;


                document.getElementById(
                    "z2"
                ).textContent =
                    data.z2;


                // ------------------------------------------------
                // Latent graph
                // ------------------------------------------------

                const graph =
                    document.getElementById(
                        "latentGraph"
                    );

                const graphPlaceholder =
                    document.getElementById(
                        "latentPlaceholder"
                    );


                graph.src =
                    "data:image/png;base64," +
                    data.latent_graph;

                graph.style.display =
                    "block";

                graphPlaceholder.style.display =
                    "none";


                // Scroll to latent graph

                document
                    .querySelector(
                        ".dark-section"
                    )
                    .scrollIntoView({
                        behavior: "smooth"
                    });


            } catch (error) {

                alert(
                    "Error: " +
                    error.message
                );

            } finally {

                button.disabled =
                    false;

                button.innerHTML =
                    `
                    <span>
                        Encode & Reconstruct
                    </span>

                    <span>
                        →
                    </span>
                    `;
            }

        }
    );


// ============================================================
// SET GENERATION COUNT
// ============================================================

function setCount(count) {

    imageCount =
        count;

    document.getElementById(
        "generationCount"
    ).textContent =
        count;


    document
        .querySelectorAll(
            ".count-buttons button"
        )
        .forEach(
            button => {

                button.classList.remove(
                    "active"
                );

                if (
                    parseInt(
                        button.textContent
                    ) === count
                ) {

                    button.classList.add(
                        "active"
                    );
                }

            }
        );
}


// ============================================================
// GENERATE NEW IMAGES
// ============================================================

document
    .getElementById(
        "generateBtn"
    )
    .addEventListener(
        "click",
        async function () {

            const button =
                this;

            button.disabled =
                true;

            button.textContent =
                "Generating...";


            try {

                const response =
                    await fetch(
                        "/generate",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify(
                                {
                                    count:
                                        imageCount
                                }
                            )
                        }
                    );


                const data =
                    await response.json();


                if (!data.success) {

                    throw new Error(
                        data.error
                    );
                }


                const grid =
                    document.getElementById(
                        "generatedGrid"
                    );


                grid.innerHTML =
                    "";


                data.images.forEach(
                    item => {

                        const card =
                            document.createElement(
                                "div"
                            );

                        card.className =
                            "generated-item";


                        card.innerHTML =
                            `
                            <img
                                src="data:image/png;base64,${item.image}"
                                alt="Generated digit"
                            >

                            <div class="generated-coordinates">
                                z₁ ${item.z1} · z₂ ${item.z2}
                            </div>
                            `;


                        grid.appendChild(
                            card
                        );

                    }
                );


            } catch (error) {

                alert(
                    "Generation error: " +
                    error.message
                );

            } finally {

                button.disabled =
                    false;

                button.textContent =
                    "✦ Generate New Images";

            }

        }
    );
