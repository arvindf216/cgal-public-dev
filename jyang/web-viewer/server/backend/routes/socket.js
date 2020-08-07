var net = require('net');

var server = net.createServer(function(socket) {
    console.log('client connected from', socket.remoteAddress, ':', socket.remotePort);
    
    // set data encoding
    socket.setEncoding('utf-8');

    // add 'data' event handler to this socket instance
    socket.on('data', (data) => {
      console.log(socket.bytesRead, 'bytes', typeof data, 'data :', data.toString('utf-8'));
    });

    socket.end('goodbye');
  }).on('error', (err) => {
    // handle errors here.
    throw err;
  });

server.listen(3001, '127.0.0.1', ()=> {
  console.log('running server on', server.address());
});

module.exports = server;