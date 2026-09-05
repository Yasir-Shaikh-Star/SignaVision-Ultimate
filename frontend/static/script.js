const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

const cameraFeed = document.getElementById("cameraFeed");

const handsStatus = document.getElementById("handsStatus");
const faceStatus = document.getElementById("faceStatus");
const poseStatus = document.getElementById("poseStatus");

const englishResult = document.getElementById("englishResult");
const urduResult = document.getElementById("urduResult");
const confidenceResult = document.getElementById("confidenceResult");
const explanationResult = document.getElementById("explanationResult");

const resultPanel = document.getElementById("resultPanel");


// =========================================================
// CAMERA
// =========================================================

let cameraStream = null;
let cameraRunning = false;
let sendingFrame = false;

let canvas = document.createElement("canvas");
let canvasContext = canvas.getContext("2d");


// =========================================================
// START BROWSER CAMERA
// =========================================================

async function startCamera() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            throw new Error(
                "Your browser does not support camera access."
            );
        }


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: "user",
                    width: {
                        ideal: 640
                    },
                    height: {
                        ideal: 480
                    }
                },

                audio: false
            });


        cameraFeed.srcObject = cameraStream;

        cameraRunning = true;


        console.log(
            "Browser camera started."
        );


        // Start sending frames
        sendCameraFrames();


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        cameraRunning = false;


        resultPanel.innerHTML = `
            <div class="waiting">
                <div class="waiting-icon">📷</div>
                <h3>Camera access required</h3>
                <p>
                    Please allow camera permission in your browser
                    to use SignaVision.
                </p>
            </div>
        `;
    }
}


// =========================================================
// SEND CAMERA FRAME TO FLASK
// =========================================================

async function sendCameraFrames() {

    if (!cameraRunning) {
        return;
    }


    // Prevent multiple requests at the same time
    if (!sendingFrame) {

        sendingFrame = true;


        try {

            if (
                cameraFeed.readyState >=
                HTMLMediaElement.HAVE_CURRENT_DATA
            ) {

                const width =
                    cameraFeed.videoWidth || 640;

                const height =
                    cameraFeed.videoHeight || 480;


                canvas.width = width;
                canvas.height = height;


                canvasContext.drawImage(
                    cameraFeed,
                    0,
                    0,
                    width,
                    height
                );


                // Convert frame to JPEG
                canvas.toBlob(
                    async function(blob) {

                        if (!blob) {

                            sendingFrame = false;

                            return;
                        }


                        try {

                            const formData =
                                new FormData();

                            formData.append(
                                "frame",
                                blob,
                                "camera.jpg"
                            );


                            const response =
                                await fetch(
                                    "/api/frame",
                                    {
                                        method: "POST",
                                        body: formData
                                    }
                                );


                            const data =
                                await response.json();


                            if (data.success &&
                                data.detection) {

                                updateDetection(
                                    data.detection
                                );
                            }


                        } catch (error) {

                            console.error(
                                "Frame upload error:",
                                error
                            );

                        } finally {

                            sendingFrame = false;
                        }

                    },
                    "image/jpeg",
                    0.70
                );

            } else {

                sendingFrame = false;
            }


        } catch (error) {

            console.error(
                "Frame processing error:",
                error
            );

            sendingFrame = false;
        }
    }


    // About 5 frames per second
    setTimeout(
        sendCameraFrames,
        200
    );
}


// =========================================================
// UPDATE DETECTION
// =========================================================

function updateDetection(data) {

    const hands =
        Number(data.hands || 0);

    const face =
        Number(data.face || 0);

    const pose =
        Number(data.pose || 0);


    handsStatus.textContent =
        `${hands} / 2`;

    faceStatus.textContent =
        `${face} / 1`;

    poseStatus.textContent =
        `${pose} / 1`;
}


// =========================================================
// ANALYZE GESTURE
// =========================================================

async function analyzeGesture() {

    if (!cameraRunning) {

        resultPanel.innerHTML = `
            <div class="waiting">
                <div class="waiting-icon">📷</div>
                <h3>Camera is not running</h3>
                <p>
                    Please allow camera access first.
                </p>
            </div>
        `;

        return;
    }


    // Loading state
    resultPanel.innerHTML = `
        <div class="waiting">
            <div class="waiting-icon">✨</div>
            <h3>Analyzing gesture...</h3>
            <p>
                AI is analyzing your hands, movement,
                face and pose.
            </p>
        </div>
    `;


    analyzeBtn.disabled = true;

    analyzeBtn.textContent =
        "✨ Analyzing...";


    try {

        const response =
            await fetch(
                "/api/analyze",
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.error ||
                "Analysis failed."
            );
        }


        displayResult(
            data.result
        );


    } catch (error) {

        console.error(
            "Analysis error:",
            error
        );


        resultPanel.innerHTML = `
            <div class="waiting">
                <div class="waiting-icon">⚠️</div>
                <h3>Analysis failed</h3>
                <p>
                    ${escapeHtml(error.message)}
                </p>
            </div>
        `;

    } finally {

        analyzeBtn.disabled = false;

        analyzeBtn.textContent =
            "✨ Analyze Gesture";
    }
}


// =========================================================
// DISPLAY AI RESULT
// =========================================================

function displayResult(result) {

    resultPanel.innerHTML = `
        <div class="result-content">

            <div class="result-main">

                <span class="result-label">
                    Detected Gesture
                </span>

                <h3>
                    ${escapeHtml(
                        result.gesture || "—"
                    )}
                </h3>

            </div>


            <div class="result-details">

                <div>
                    <span>Hands</span>
                    <strong>
                        ${escapeHtml(
                            result.hands || "—"
                        )}
                    </strong>
                </div>

                <div>
                    <span>Movement</span>
                    <strong>
                        ${escapeHtml(
                            result.movement || "—"
                        )}
                    </strong>
                </div>

                <div>
                    <span>Face</span>
                    <strong>
                        ${escapeHtml(
                            result.face || "—"
                        )}
                    </strong>
                </div>

                <div>
                    <span>Pose</span>
                    <strong>
                        ${escapeHtml(
                            result.pose || "—"
                        )}
                    </strong>
                </div>

                <div>
                    <span>Possible PSL Meaning</span>
                    <strong>
                        ${escapeHtml(
                            result.possible_psl_meaning ||
                            "—"
                        )}
                    </strong>
                </div>

            </div>

        </div>
    `;


    englishResult.textContent =
        result.english || "—";

    urduResult.textContent =
        result.urdu || "—";

    confidenceResult.textContent =
        result.confidence || "—";

    explanationResult.textContent =
        result.explanation || "—";
}


// =========================================================
// CLEAR RESULT
// =========================================================

function clearResult() {

    resultPanel.innerHTML = `
        <div class="waiting">

            <div class="waiting-icon">
                🤖
            </div>

            <h3>
                Waiting for a gesture
            </h3>

            <p>
                Perform a sign and click
                <strong>Analyze Gesture</strong>.
            </p>

        </div>
    `;


    englishResult.textContent = "—";
    urduResult.textContent = "—";
    confidenceResult.textContent = "—";

    explanationResult.textContent =
        "The AI explanation will appear here after gesture analysis.";
}


// =========================================================
// HTML ESCAPE
// =========================================================

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value);

    return div.innerHTML;
}


// =========================================================
// BUTTON EVENTS
// =========================================================

analyzeBtn.addEventListener(
    "click",
    analyzeGesture
);

clearBtn.addEventListener(
    "click",
    clearResult
);


// =========================================================
// START CAMERA WHEN PAGE LOADS
// =========================================================

window.addEventListener(
    "load",
    startCamera
);


// =========================================================
// STOP CAMERA WHEN PAGE CLOSES
// =========================================================

window.addEventListener(
    "beforeunload",
    function() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );
        }
    }
);