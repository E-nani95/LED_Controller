package com.example.myapplication;
// TcpClient.kt 파일
import kotlinx.coroutines.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.Socket

class TcpClient(
    private val host: String,
    private val port: Int,
    private val scope: CoroutineScope // 액티비티의 생명주기를 따르는 코루틴 스코프
) {
    private var socket: Socket? = null
    private var writer: PrintWriter? = null
    private var reader: BufferedReader? = null

    // 메시지 수신 시 호출될 리스너 (콜백)
    var onMessageReceived: ((String) -> Unit)? = null

    fun connect() {
        // 네트워크 작업은 반드시 백그라운드 스레드에서 수행
        scope.launch(Dispatchers.IO) {
            try {
                socket = Socket(host, port)
                writer = PrintWriter(socket!!.outputStream, true)
                reader = BufferedReader(InputStreamReader(socket!!.inputStream))

                // 연결 성공 후 메시지 수신 시작
                startListening()

            } catch (e: Exception) {
                // UI 스레드로 전환하여 에러 메시지 전달
                withContext(Dispatchers.Main) {
                    onMessageReceived?.invoke("Error connecting: ${e.message}")
                }
            }
        }
    }

    private fun startListening() {
        // 메시지 수신 또한 백그라운드 스레드에서 계속 수행
        scope.launch(Dispatchers.IO) {
            try {
                while (isActive) { // 코루틴이 활성 상태인 동안 계속 실행
                    val message = reader?.readLine()
                    if (message == null) {
                        // 서버가 연결을 끊음
                        withContext(Dispatchers.Main) {
                            onMessageReceived?.invoke("Server disconnected.")
                        }
                        break
                    } else {
                        // UI 스레드로 전환하여 수신된 메시지 전달
                        withContext(Dispatchers.Main) {
                            onMessageReceived?.invoke(message)
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onMessageReceived?.invoke("Error receiving: ${e.message}")
                }
            }
        }
    }

    fun sendMessage(message: String) {
        // 메시지 전송도 백그라운드 스레드에서
        scope.launch(Dispatchers.IO) {
            try {
                writer?.println(message)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onMessageReceived?.invoke("Error sending: ${e.message}")
                }
            }
        }
    }

    fun close() {
        scope.launch(Dispatchers.IO) {
            try {
                socket?.close()
                writer?.close()
                reader?.close()
            } catch (e: Exception) {
                // Log the error or handle it as needed
            }
        }
    }
}