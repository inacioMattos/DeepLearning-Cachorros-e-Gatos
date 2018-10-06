const remote = require("electron").remote;

let janela = remote.getCurrentWindow();


var zerorpc = require("zerorpc");

var client = new zerorpc.Client();
client.connect("tcp://localhost:8080");



function quit()
{
	janela.close();
}


function mini()
{
	janela.minimize();
}



function sendPython()
{
	let txt = document.getElementsByClassName("txt")[0].value;
	let out = document.getElementsByClassName("txtOut")[0];


	client.invoke("echo", txt, (error, res) => {
		if(error) {
		  console.error(error);
		} else {
		  out.innerHTML = res;
		}
	});
}