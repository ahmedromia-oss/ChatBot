const msgerForm = get(".msger-inputarea");
const msgerInput = get("#msger-input");
const msgerChat = get(".msger-chat");

var socket;
var firstMessage = ""

var UserName = ""


$(document).ready(function(){
    socket = io.connect('http://' + document.domain + ':' + location.port + '/chat');
    socket.on('connect', function() {
      console.log("You are connected")
        socket.emit('joined', {});
    });
    socket.on('status', function(data) {
        firstMessage = data.msg
    });
    socket.on('message', function(data) {
    
    appendMessage(Username ,PERSON_IMG, "right", data.msg);
    if((data.msg).toLowerCase() == "quit"){
      
        socket.emit('left', {}, function() {
            socket.disconnect();

            // go back to the login page
            window.location.href = "http://127.0.0.1:5000/";
    
        }
    
        );
    }
    else if((data.msg).toLowerCase() == "listen"){
      appendMessage("BOT", "static/images/BotImg.png", "left", "Listening.....");
    }


    });

    socket.on('messageBot', function(data) {
      botResponse(data.msg)

        
    }); 
    $("#msger-send-btn").click(
      function(){
        text = $('#msger-input').val();
            $('#msger-input').val('');
            socket.emit('text', {msg: text});
            socket.emit('botResponse' , {msg:text})
      });
    $('#msger-input').keypress(function(e) {
        var code = e.keyCode || e.which;
        if (code == 13) {
            text = $('#msger-input').val();
            $('#msger-input').val('');
            socket.emit('text', {msg: text});
            socket.emit('botResponse' , {msg:text})}
    });
});
function leave_room() {
    socket.emit('left', {}, function() {
        socket.disconnect();
        window.location.href = "{{ url_for('main.index') }}";
    });
}


// Default Data
const BOT_IMG = "static/images/BotImg.png";
const PERSON_IMG = "static/images/userImg.png";
const BOT_NAME = "BOT";

function firstMssg() {
  
    const messg = firstMessage
    Username = (messg.split('!'))[0]

    appendMessage(BOT_NAME, BOT_IMG, "left", messg);
}
window.onload = setTimeout(()=>firstMssg(), 2000);

function appendMessage(name, img, side, text) {
  const msgHTML = `
    <div class="msg ${side}-msg">
      <div class="msg-img" style="background-image: url(${img})"></div>

      <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">${name}</div>
          <div class="msg-info-time">${formatDate(new Date())}</div>
        </div>

        <div class="msg-text">${text}</div>
      </div>
    </div>
  `;

  msgerChat.insertAdjacentHTML("beforeend", msgHTML);   
  msgerChat.scrollTop += 500;
}

function botResponse(message) {
  
  const delay = message.split(" ").length * 100;

  setTimeout(() => {
    appendMessage(BOT_NAME, BOT_IMG, "left", message);
  }, delay);
}

// Utils
function get(selector, root = document) {
  return root.querySelector(selector);
}

function formatDate(date) {
  const h = "0" + date.getHours();
  const m = "0" + date.getMinutes();

  return `${h.slice(-2)}:${m.slice(-2)}`;
}

