package com.example.myapplication;
// MainActivity.kt 파일
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope // lifecycleScope를 사용하기 위해 추가
import com.example.myapplication.R
import android.util.Log
import androidx.appcompat.widget.SwitchCompat
class MainActivity : AppCompatActivity() {
    private lateinit var tcpClient: TcpClient
    private lateinit var txt1: TextView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        val btn1 =findViewById<SwitchCompat>(R.id.btn1)
        val txt1=findViewById<TextView>(R.id.txt1)
        try {
            tcpClient = TcpClient("192.168.0.49", 7777, lifecycleScope)
//        tcpClient = TcpClient("10.0.2.2", 7777, lifecycleScope)

            tcpClient.onMessageReceived = { message ->
                // 이 람다는 TcpClient 내부에서 Main 스레드로 전환되어 호출되므로
                // 안전하게 UI를 업데이트할 수 있습니다.
                Log.d("TcpClient", "Message Received: '$message'")
                txt1.text = message
            }
        }catch (e: Exception){
            Log.e("TcpClient", "연결 실패: ${e.message}", e)
        }
        tcpClient.connect()


        btn1.setOnCheckedChangeListener { _, isChecked ->
            // isChecked 변수는 스위치의 '새로운' 상태를 직접 알려줍니다.
            // true이면 ON, false이면 OFF 입니다.

            val messageToSend = if (isChecked) {
                // 스위치가 ON 상태가 되었으므로 "on" 메시지를 보냅니다.
                "on"
            } else {
                // 스위치가 OFF 상태가 되었으므로 "off" 메시지를 보냅니다.
                "off"
            }

            /*
             * UI 업데이트(txt1.text = ...)는 여기서 바로 하기보다,
             * 서버로부터 "ON", "OFF" 응답을 받았을 때 (onMessageReceived 에서)
             * 처리하는 것이 더 안정적인 구조입니다.
             * 지금은 일단 주석 처리하거나 삭제하는 것을 권장합니다.
             */
            // if (isChecked) {
            //     txt1.text = "ON"
            // } else {
            //     txt1.text = "OFF"
            // }

            // 서버로 결정된 메시지("on" 또는 "off")를 전송합니다.
            Log.d("Switch", "Sending message to server: $messageToSend")
            tcpClient.sendMessage(messageToSend)
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        // 5. 액티비티가 종료될 때 TCP 연결도 반드시 닫아줍니다.
        tcpClient.close()
    }
}



fun testFunc(): String{
    System.out.println("test");
    return "function test";
}

