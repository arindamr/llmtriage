import React, { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);

    const endpoint = sessionId ? "/api/continue" : "/api/triage";
    const payload = sessionId
      ? { session_id: sessionId, additional_input: input }
      : { problem: input };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (data.error) {
        setMessages([...newMessages, { role: "error", content: data.error }]);
      } else {
        setSessionId(data.session_id);
        setMessages([
          ...newMessages,
          { role: "assistant", content: data.state.final_diagnosis },
        ]);
      }
    } catch (error) {
      setMessages([
        ...newMessages,
        { role: "error", content: "An error occurred. Please try again." },
      ]);
    }

    setInput("");
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.role === "user" ? "user" : "assistant"}`}
            >
              {msg.content}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default App;
