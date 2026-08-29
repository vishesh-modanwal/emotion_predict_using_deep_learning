// ============================================================
// EMOTISENSE FRONTEND
// ============================================================


// ============================================================
// DOM ELEMENTS
// ============================================================

const input = document.getElementById("emotionInput");

const charCount = document.getElementById("charCount");

const analyzeButton =
    document.getElementById("analyzeButton");

const buttonLoader =
    document.getElementById("buttonLoader");

const clearButton =
    document.getElementById("clearButton");

const newAnalysisButton =
    document.getElementById("newAnalysisButton");

const emptyState =
    document.getElementById("emptyState");

const resultContent =
    document.getElementById("resultContent");

const emotionName =
    document.getElementById("emotionName");

const emotionEmoji =
    document.getElementById("emotionEmoji");

const emotionDescription =
    document.getElementById("emotionDescription");

const confidenceText =
    document.getElementById("confidenceText");

const confidenceBadge =
    document.getElementById("confidenceBadge");

const confidenceBar =
    document.getElementById("confidenceBar");

const probabilityList =
    document.getElementById("probabilityList");

const modelStatus =
    document.getElementById("modelStatus");

const systemStatus =
    document.getElementById("systemStatus");

const toast =
    document.getElementById("toast");

const toastMessage =
    document.getElementById("toastMessage");

const toastIcon =
    document.getElementById("toastIcon");


// ============================================================
// EMOTION INFORMATION
// ============================================================

const emotionInfo = {

    sadness: {
        emoji: "😔",
        description:
            "A feeling of sadness, loss, disappointment, or low mood."
    },

    joy: {
        emoji: "😊",
        description:
            "A positive emotional state associated with happiness and excitement."
    },

    love: {
        emoji: "💕",
        description:
            "A warm emotional connection involving affection, care, or attachment."
    },

    fear: {
        emoji: "😥",
        description:
            "A response associated with worry, uncertainty, threat, or anxiety."
    },

    surprise: {
        emoji: "😎",
        description:
            "An emotional response to something unexpected or remarkable."
    },

    anger: {
        emoji: "😡",
        description:
            "A strong emotional response associated with frustration or injustice."
    }

};


// ============================================================
// CHARACTER COUNTER
// ============================================================

input.addEventListener("input", () => {

    const length = input.value.length;

    charCount.textContent = length;

});


// ============================================================
// EXAMPLE CHIPS
// ============================================================

document
    .querySelectorAll(".example-chip")
    .forEach(button => {

        button.addEventListener("click", () => {

            input.value =
                button.dataset.text;

            input.dispatchEvent(
                new Event("input")
            );

            input.focus();

        });

    });


// ============================================================
// CLEAR INPUT
// ============================================================

clearButton.addEventListener(
    "click",
    clearInput
);


function clearInput() {

    input.value = "";

    input.dispatchEvent(
        new Event("input")
    );

    input.focus();

}


// ============================================================
// NEW ANALYSIS
// ============================================================

newAnalysisButton.addEventListener(
    "click",
    () => {

        resultContent.classList.add("hidden");

        emptyState.classList.remove("hidden");

        clearInput();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);


// ============================================================
// KEYBOARD SHORTCUT
// ============================================================

input.addEventListener(
    "keydown",
    event => {

        if (
            (event.ctrlKey || event.metaKey)
            &&
            event.key === "Enter"
        ) {

            event.preventDefault();

            analyzeEmotion();

        }

    }
);


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzeEmotion
);


// ============================================================
// ANALYZE EMOTION
// ============================================================

async function analyzeEmotion() {

    const text = input.value.trim();


    // --------------------------------------------------------
    // Validate input
    // --------------------------------------------------------

    if (!text) {

        showToast(
            "Please write something first.",
            "!"
        );

        input.focus();

        return;
    }


    if (text.length > 2000) {

        showToast(
            "Text must be under 2000 characters.",
            "!"
        );

        return;
    }


    // --------------------------------------------------------
    // Loading state
    // --------------------------------------------------------

    setLoading(true);


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );


        const data = await response.json();


        // ----------------------------------------------------
        // Handle API errors
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to analyze the text."
            );

        }


        // ----------------------------------------------------
        // Display result
        // ----------------------------------------------------

        displayResult(data);


    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        showToast(
            error.message ||
            "Something went wrong. Please try again.",
            "!"
        );

    }

    finally {

        setLoading(false);

    }

}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    // --------------------------------------------------------
    // Extract emotion
    // --------------------------------------------------------

    let detectedEmotion =
        data.predicted_emotion
            .toLowerCase()
            .trim();


    // Remove emoji if API returns it
    detectedEmotion =
        detectedEmotion
            .replace(
                /[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu,
                ""
            )
            .trim();


    // --------------------------------------------------------
    // Find emotion information
    // --------------------------------------------------------

    const info =
        emotionInfo[detectedEmotion]
        ||
        {
            emoji: "✦",
            description:
                "The model identified an emotional signal in your text."
        };


    // --------------------------------------------------------
    // Update emotion
    // --------------------------------------------------------

    emotionName.textContent =
        detectedEmotion;


    emotionEmoji.textContent =
        info.emoji;


    emotionDescription.textContent =
        info.description;


    // --------------------------------------------------------
    // Confidence
    // --------------------------------------------------------

    const confidence =
        Number(data.confidence) * 100;


    const roundedConfidence =
        Math.round(confidence);


    confidenceText.textContent =
        `${roundedConfidence}%`;


    confidenceBadge.textContent =
        `${roundedConfidence}%`;


    // Reset progress bar first
    confidenceBar.style.width = "0%";


    // Animate after browser paint
    requestAnimationFrame(() => {

        setTimeout(() => {

            confidenceBar.style.width =
                `${confidence}%`;

        }, 80);

    });


    // --------------------------------------------------------
    // Probability breakdown
    // --------------------------------------------------------

    renderProbabilities(
        data.all_probability
    );


    // --------------------------------------------------------
    // Switch empty state -> result
    // --------------------------------------------------------

    emptyState.classList.add("hidden");

    resultContent.classList.remove("hidden");


    // --------------------------------------------------------
    // Scroll result into view on mobile
    // --------------------------------------------------------

    if (window.innerWidth < 850) {

        setTimeout(() => {

            document
                .getElementById("resultCard")
                .scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

        }, 250);

    }

}


// ============================================================
// RENDER PROBABILITIES
// ============================================================

function renderProbabilities(probabilities) {

    probabilityList.innerHTML = "";


    // Convert object into array
    const entries =
        Object.entries(probabilities);


    // Sort highest -> lowest
    entries.sort(
        (a, b) => b[1] - a[1]
    );


    entries.forEach(
        ([emotion, probability], index) => {

            const percentage =
                Number(probability) * 100;


            const row =
                document.createElement("div");

            row.className =
                "probability-row";


            row.innerHTML = `

                <span class="probability-label">
                    ${emotion}
                </span>

                <div class="probability-track">

                    <div
                        class="probability-fill"
                        data-width="${percentage}"
                        style="transition-delay: ${index * 70}ms"
                    ></div>

                </div>

                <span class="probability-value">
                    ${Math.round(percentage)}%
                </span>

            `;


            probabilityList.appendChild(row);

        }
    );


    // Animate bars
    requestAnimationFrame(() => {

        setTimeout(() => {

            document
                .querySelectorAll(
                    ".probability-fill"
                )
                .forEach(bar => {

                    bar.style.width =
                        `${bar.dataset.width}%`;

                });

        }, 100);

    });

}


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    if (isLoading) {

        analyzeButton.classList.add(
            "loading"
        );

        analyzeButton.disabled = true;

    }

    else {

        analyzeButton.classList.remove(
            "loading"
        );

        analyzeButton.disabled = false;

    }

}


// ============================================================
// TOAST
// ============================================================

let toastTimer;


function showToast(
    message,
    icon = "!"
) {

    toastMessage.textContent =
        message;

    toastIcon.textContent =
        icon;

    toast.classList.add("show");


    clearTimeout(toastTimer);


    toastTimer =
        setTimeout(() => {

            toast.classList.remove(
                "show"
            );

        }, 3500);

}


// ============================================================
// HEALTH CHECK
// ============================================================

async function checkServerHealth() {

    try {

        const response =
            await fetch("/health");


        if (!response.ok) {

            throw new Error(
                "Server unavailable"
            );

        }


        const data =
            await response.json();


        if (data.model_loaded) {

            modelStatus.textContent =
                "Model Ready";

            systemStatus.textContent =
                "AI System Online";

        }

        else {

            modelStatus.textContent =
                "Model Offline";

            systemStatus.textContent =
                "Model Unavailable";

        }

    }

    catch (error) {

        console.error(
            "Health check failed:",
            error
        );


        modelStatus.textContent =
            "Server Offline";

        systemStatus.textContent =
            "Connection Error";

    }

}


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        checkServerHealth();

    }
);