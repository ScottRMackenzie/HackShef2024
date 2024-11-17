package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

var emergency = false
var reportStartTime time.Time = time.Now().AddDate(200, 0, 0) // 200 years from now

func main() {
	// create api mux on port 2113
	mux := http.NewServeMux()

	mux.HandleFunc("/report", func(w http.ResponseWriter, r *http.Request) {
		go handleReport()
		w.Write([]byte("Emergency reported. Please evacuate the area."))
		return
	})

	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{"Emergency": emergency, "ReportStartTime": reportStartTime})
		w.Write([]byte(fmt.Sprintf("Emergency Status: %t", emergency)))
	})

	mux.HandleFunc("/cancel", func(w http.ResponseWriter, r *http.Request) {
		emergency = false
		reportStartTime = time.Now().AddDate(200, 0, 0)
		w.Write([]byte("Emergency cancelled."))
	})

	// Start the server on port 2113
	fmt.Println("Server starting on :2113")
	err := http.ListenAndServe(":2113", mux)
	if err != nil {
		panic(err)
	}

	
}

func handleReport() {
	fmt.Println("Emergency reported. Please evacuate the area.")
	reportStartTime = time.Now()

	// wait for a minute
	time.Sleep(1 * time.Minute)

	// if the report is still active, play the sfx
	if time.Now().Sub(reportStartTime) < 1*time.Minute {
		fmt.Println("Deploying Countermeasures")
	}
	return
}
