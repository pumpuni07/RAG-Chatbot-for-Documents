/**
 * script.js — Chatbot Frontend Logic
 * ====================================
 * IBM Generative AI Engineering Professional Certificate
 * Project: Build a Chatbot for Your Data
 *
 * Author  : Jack Pumpuni Frimpong-Manso
 * Date    : 2026
 * License : Apache 2.0
 *
 * Responsibilities:
 *   - Send user messages to /process-message and display responses
 *   - Handle PDF file uploads to /process-document
 *   - Show / hide loading animation during server processing
 *   - Maintain light / dark mode toggle
 *   - Reset chat history via /reset endpoint
 *   - Sanitise all user inputs before sending
 */

"use strict";

// ── Configuration ─────────────────────────────────────────────────────────────
const baseUrl      = "";      // same origin — Flask serves both frontend and API
let   lightMode    = true;    // tracks current theme
let   isFirstMessage = true;  // disables send until a document is uploaded


// ── Utility: scroll chat to bottom ───────────────────────────────────────────
function scrollToBottom() {
  const list = document.getElementById("message-list");
  list.scrollTop = list.scrollHeight;
}


// ── Utility: sleep ────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));


// ── Utility: sanitise text input ─────────────────────────────────────────────
/**
 * Removes leading/trailing whitespace, newlines, tabs, and HTML tags
 * to prevent XSS and malformed requests.
 */
function cleanTextInput(value) {
  return value
    .trim()
    .replace(/[\n\t]/g, "")
    .replace(/<[^>]*>/g, "")
    .replace(/[<>&;]/g, "");
}


// ── Loading animation ─────────────────────────────────────────────────────────
async function showBotLoadingAnimation() {
  await sleep(200);
  $(".loading-animation").css("display", "flex");
  $("#send-button").prop("disabled", true);
  scrollToBottom();
}

function hideBotLoadingAnimation() {
  $(".loading-animation").hide();
  if (!isFirstMessage) {
    $("#send-button").prop("disabled", false);
  }
}


// ── Append a user message bubble ──────────────────────────────────────────────
function populateUserMessage(userMessage) {
  $("#message-input").val("");
  $("#message-list").append(`
    <div class="message-line my-text">
      <div class="message-box${!lightMode ? " dark" : ""}">
        ${userMessage}
      </div>
    </div>
  `);
  scrollToBottom();
}


// ── Append a bot response bubble ──────────────────────────────────────────────
function populateBotResponse(responseText) {
  // Provide an upload shortcut inside the response bubble on first message
  const uploadHtml = isFirstMessage
    ? `<br/><label for="file-upload" class="upload-btn-chat">
         <i class="fa-solid fa-file-arrow-up me-1"></i>Upload PDF
       </label>`
    : "";

  $("#message-list").append(`
    <div class="message-line">
      <div class="message-box bot-message${!lightMode ? " dark" : ""}">
        <i class="fa-solid fa-robot me-2" style="color:#0984e3;font-size:0.85rem;"></i>
        ${responseText}${uploadHtml}
      </div>
    </div>
  `);
  scrollToBottom();
}


// ── Send a text message to the server ────────────────────────────────────────
async function processUserMessage(userMessage) {
  let response = await fetch(baseUrl + "/process-message", {
    method:  "POST",
    headers: { "Accept": "application/json", "Content-Type": "application/json" },
    body:    JSON.stringify({ userMessage }),
  });
  response = await response.json();
  return response;
}


// ── Handle send button / Enter key ───────────────────────────────────────────
async function handleSend() {
  const raw = $("#message-input").val();
  const userMessage = cleanTextInput(raw);

  if (!userMessage) return;

  populateUserMessage(userMessage);
  await showBotLoadingAnimation();

  try {
    const response = await processUserMessage(userMessage);
    hideBotLoadingAnimation();
    populateBotResponse(response.botResponse || "Sorry, I could not generate a response.");
  } catch (err) {
    hideBotLoadingAnimation();
    populateBotResponse("⚠️ Error communicating with the server. Please try again.");
    console.error("Error sending message:", err);
  }
}

// Send on button click
$("#send-button").on("click", handleSend);

// Send on Enter key (not Shift+Enter)
$("#message-input").on("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});


// ── Handle PDF file upload ────────────────────────────────────────────────────
$("#file-upload").on("change", async function () {
  const file = this.files[0];
  if (!file) return;

  if (file.type !== "application/pdf") {
    populateBotResponse("⚠️ Please upload a PDF file only.");
    return;
  }

  populateBotResponse(
    `<i class="fa-solid fa-spinner fa-spin me-2"></i>
     Processing <strong>${file.name}</strong> — please wait...`
  );
  await showBotLoadingAnimation();

  const reader = new FileReader();

  reader.onload = async function (e) {
    try {
      let response = await fetch(baseUrl + "/process-document", {
        method:  "POST",
        headers: { "Accept": "application/json", "Content-Type": "application/json" },
        body:    JSON.stringify({ fileData: e.target.result }),
      });
      response = await response.json();

      hideBotLoadingAnimation();

      // Enable the send button now that a document is loaded
      isFirstMessage = false;
      $("#send-button").prop("disabled", false);

      populateBotResponse(
        response.botResponse ||
        "Document processed! You can now ask questions about it."
      );
    } catch (err) {
      hideBotLoadingAnimation();
      populateBotResponse("⚠️ Error uploading document. Please try again.");
      console.error("Error uploading document:", err);
    }
  };

  reader.onerror = function () {
    hideBotLoadingAnimation();
    populateBotResponse("⚠️ Could not read the file. Please try again.");
  };

  reader.readAsDataURL(file);

  // Reset file input so the same file can be re-uploaded if needed
  this.value = "";
});


// ── Reset chat ────────────────────────────────────────────────────────────────
$("#reset-button").on("click", async function () {
  // Clear chat UI
  $("#message-list").empty();

  // Reset server-side chat history
  try {
    await fetch(baseUrl + "/reset", { method: "POST" });
  } catch (err) {
    console.warn("Could not reset server chat history:", err);
  }

  // Reset state
  isFirstMessage = true;
  $("#send-button").prop("disabled", true);

  // Show greeting again
  populateBotResponse(
    "Chat reset! Please upload a new PDF document to start a fresh conversation."
  );
});


// ── Light / Dark mode toggle ──────────────────────────────────────────────────
$("#light-dark-mode-switch").on("change", function () {
  $("body").toggleClass("dark-mode");
  $(".message-box").toggleClass("dark");
  $(".loading-dots").toggleClass("dark");
  $(".dot").toggleClass("dark-dot");
  lightMode = !lightMode;
});


// ── On page load: disable send until document is uploaded ────────────────────
$(document).ready(function () {
  $("#send-button").prop("disabled", true);
  scrollToBottom();
});
