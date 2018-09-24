const remote = require("electron").remote;

let janela = remote.getCurrentWindow();



function quit()
{
	janela.close();
}


function mini()
{
	janela.minimize();
}