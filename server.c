#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <wiringPi.h>

#define BUFFSIZE 200
#define LED_PIN 23


void send_message(char * msg, int my_sock){
	write(my_sock,msg,strlen(msg));
}


void* clnt_connection(void * arg){

	int clnt_sock = (int)arg;
	int str_len=0;

	char msg[BUFFSIZE];
	int i;
	
	char work_result[10];



	while(1){
		str_len = read(clnt_sock,msg,sizeof(msg));
        // if(str_len > 0) {
        //     msg[str_len] = '\0';
        // }
        msg[str_len] = '\0';
		if(str_len == -1){
			printf("clnt[%d] close\n",clnt_sock);
			break;
		}
        if (msg[str_len - 1] == '\n') {
            msg[str_len - 1] = '\0';
        }
        // msg[str_len] = '\0';
		if (strcmp(msg,"on")==0){
            printf("on");
			// snprintf(work_result, sizeof(work_result),"ON\n");
			snprintf(work_result, sizeof(work_result),"ON\n");
			digitalWrite(LED_PIN, HIGH);
		}
		else if(strcmp(msg,"off")==0){
            printf("off");
			// snprintf(work_result, sizeof(work_result),"OFF\n");
			snprintf(work_result, sizeof(work_result),"OFF\n");
			digitalWrite(LED_PIN, LOW);
		}else{
            printf("fail");
			// snprintf(work_result, sizeof(work_result),"fail\n");
			snprintf(work_result, sizeof(work_result),"fail");
		}
		printf("before send\n");
		send_message(work_result,clnt_sock);
		printf("%s\n",msg);
	}




}


int main(){
	int serv_sock;
	int clnt_sock;
	
	pthread_t t_thread;

	int clnt_addr_size;

	struct sockaddr_in clnt_addr;
	struct sockaddr_in serv_addr;

        	// BCM way
	wiringPiSetupGpio();

	pinMode(LED_PIN, OUTPUT);

	serv_sock=socket(PF_INET,SOCK_STREAM,0);
	serv_addr.sin_family=AF_INET;
	serv_addr.sin_addr.s_addr=htonl(INADDR_ANY);
	serv_addr.sin_port=htons(7777);


	if(bind(serv_sock,(struct sockaddr*)&serv_addr,sizeof(serv_addr))==-1){
		printf("bind error\n");
	}

	if(listen(serv_sock,5)==-1){
		printf("listen error");
	}
	
	char buff[300];
	int recv_len=0;

	while(1){
		clnt_addr_size=sizeof(clnt_addr);
		clnt_sock=accept(serv_sock,(struct sockaddr*)&clnt_addr,&clnt_addr_size);
		pthread_create(&t_thread,NULL,clnt_connection,(void *)clnt_sock);
		//while(1){
		//recv_len=read(clnt_sock,buff,300);
		//printf("recv: %s \n",buff);
		//}
	}
	
}
