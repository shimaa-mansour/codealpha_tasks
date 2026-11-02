"""Simple web chat UI built with Flask."""
from flask import Flask, jsonify, render_template_string, request

from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()

PAGE = """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>FAQ Chatbot</title>
<style>
  body{font-family:Arial,sans-serif;background:#f0f2f5;display:flex;justify-content:center;padding:30px}
  .box{width:420px;background:#fff;border-radius:10px;box-shadow:0 2px 10px #0002;display:flex;flex-direction:column;height:560px}
  h2{margin:0;padding:14px;background:#2563eb;color:#fff;border-radius:10px 10px 0 0}
  #chat{flex:1;overflow-y:auto;padding:14px}
  .msg{margin:8px 0;padding:8px 12px;border-radius:14px;max-width:80%}
  .user{background:#2563eb;color:#fff;margin-left:auto}
  .bot{background:#e5e7eb}
  form{display:flex;border-top:1px solid #ddd}
  input{flex:1;border:0;padding:14px;outline:0}
  button{border:0;background:#2563eb;color:#fff;padding:0 18px;cursor:pointer}
</style></head>
<body><div class="box">
  <h2>FAQ Chatbot</h2>
  <div id="chat"><div class="msg bot">Hi! Ask me anything about the internship.</div></div>
  <form id="f"><input id="q" placeholder="Type your question..." autocomplete="off"><button>Send</button></form>
</div>
<script>
const chat=document.getElementById('chat'), f=document.getElementById('f'), q=document.getElementById('q');
function add(text,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
f.addEventListener('submit',async e=>{
  e.preventDefault(); const text=q.value.trim(); if(!text) return;
  add(text,'user'); q.value='';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
  add((await r.json()).answer,'bot');
});
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.get_json().get("question", "")
    return jsonify(answer=bot.get_answer(question))


if __name__ == "__main__":
    app.run(debug=True)
