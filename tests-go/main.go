package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"github.com/chromedp/chromedp"
)

// PatentData holds structured patent information
type PatentData struct {
	Title           string
	PublicationNum  string
	FilingDate      string
	PublicationDate string
	Inventors       []string
	Assignee        string
	Abstract        string
	Claims          []string
	Description     string
	Classifications []string
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <patent-url>")
	}

	patentURL := os.Args[1]
	data, err := scrapePatent(patentURL)
	if err != nil {
		log.Fatalf("Scraping failed: %v", err)
	}

	markdown := generateLLMMarkdown(data)
	fmt.Println(markdown)
}

func scrapePatent(url string) (*PatentData, error) {
	ctx, cancel := chromedp.NewContext(context.Background())
	defer cancel()

	var htmlContent string
	err := chromedp.Run(ctx,
		chromedp.Navigate(url),
		chromedp.WaitVisible(`section[itemtype="http://schema.org/Patent"]`, chromedp.ByQuery),
		chromedp.OuterHTML(`document.documentElement`, &htmlContent, chromedp.ByJSPath),
	)
	if err != nil {
		return nil, fmt.Errorf("navigation failed: %w", err)
	}

	doc, err := goquery.NewDocumentFromReader(strings.NewReader(htmlContent))
	if err != nil {
		return nil, fmt.Errorf("parsing failed: %w", err)
	}

	data := &PatentData{}

	// Title
	data.Title = strings.TrimSpace(doc.Find(`span[itemprop="title"]`).First().Text())
	data.PublicationNum = extractPatentNumber(url)

	// Metadata
	doc.Find(`dl[itemprop="application"] dd`).Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())
		switch i {
		case 0:
			data.FilingDate = text
		case 1:
			data.PublicationDate = text
		}
	})

	// Inventors
	doc.Find(`dd[itemprop="inventor"]`).Each(func(_ int, s *goquery.Selection) {
		data.Inventors = append(data.Inventors, strings.TrimSpace(s.Text()))
	})

	// Assignee
	data.Assignee = strings.TrimSpace(doc.Find(`span[itemprop="assigneeCurrent"]`).Text())

	// Abstract
	data.Abstract = strings.TrimSpace(doc.Find(`section[itemprop="abstract"] div`).Text())

	// Claims
	doc.Find(`div[itemprop="claims"] div.claim`).Each(func(_ int, s *goquery.Selection) {
		claimNum := strings.TrimSpace(s.Find(`div.claim-num`).Text())
		claimText := strings.TrimSpace(s.Find(`div.claim-text`).Text())
		data.Claims = append(data.Claims, fmt.Sprintf("%s %s", claimNum, claimText))
	})

	// Description (full text)
	data.Description = strings.TrimSpace(doc.Find(`section[itemprop="description"]`).Text())

	// Classifications
	doc.Find(`span[itemprop="cpc"]`).Each(func(_ int, s *goquery.Selection) {
		data.Classifications = append(data.Classifications, strings.TrimSpace(s.Text()))
	})

	return data, nil
}

func extractPatentNumber(url string) string {
	re := regexp.MustCompile(`/patent/([A-Z0-9]+)`)
	matches := re.FindStringSubmatch(url)
	if len(matches) > 1 {
		return matches[1]
	}
	return ""
}

func generateLLMMarkdown(data *PatentData) string {
	var sb strings.Builder

	// H1: Patent identifier (concise, unique)
	sb.WriteString(fmt.Sprintf("# %s - %s\n\n", data.PublicationNum, data.Title))

	// Metadata table (dense, machine-readable)
	sb.WriteString("## Patent Metadata\n\n")
	sb.WriteString("| Field | Value |\n")
	sb.WriteString("|-------|-------|\n")
	sb.WriteString(fmt.Sprintf("| Publication Number | %s |\n", data.PublicationNum))
	sb.WriteString(fmt.Sprintf("| Filing Date | %s |\n", data.FilingDate))
	sb.WriteString(fmt.Sprintf("| Publication Date | %s |\n", data.PublicationDate))
	sb.WriteString(fmt.Sprintf("| Inventors | %s |\n", strings.Join(data.Inventors, "; ")))
	sb.WriteString(fmt.Sprintf("| Assignee | %s |\n\n", data.Assignee))

	// Abstract (single block)
	sb.WriteString("## Abstract\n\n")
	sb.WriteString(data.Abstract + "\n\n")

	// Claims (code block for structured parsing)
	sb.WriteString("## Claims\n\n```patent\n")
	for _, claim := range data.Claims {
		sb.WriteString(claim + "\n")
	}
	sb.WriteString("```\n\n")

	// Description (preserve hierarchy but flatten where possible)
	sb.WriteString("## Description\n\n")
	// Clean up excessive whitespace
	desc := regexp.MustCompile(`\n\s*\n`).ReplaceAllString(data.Description, "\n\n")
	sb.WriteString(desc + "\n\n")

	// Classifications (bullet list, atomic)
	sb.WriteString("## Classifications\n\n")
	for _, class := range data.Classifications {
		sb.WriteString(fmt.Sprintf("- %s\n", class))
	}

	return sb.String()
}
