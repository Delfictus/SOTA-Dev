//! AI Dialogue System for PRISM
//!
//! Conversational interface with intelligent responses.

/// A message in the dialogue
#[derive(Debug, Clone)]
pub struct Message {
    pub content: String,
    pub is_user: bool,
    pub timestamp: std::time::Instant,
}

/// AI Dialogue manager
pub struct AiDialogue {
    pub messages: Vec<Message>,
    pub scroll_offset: usize,
}

impl AiDialogue {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            scroll_offset: 0,
        }
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message {
            content: content.to_string(),
            is_user: true,
            timestamp: std::time::Instant::now(),
        });
    }

    /// Add a system/AI message
    pub fn add_system_message(&mut self, content: &str) {
        self.messages.push(Message {
            content: content.to_string(),
            is_user: false,
            timestamp: std::time::Instant::now(),
        });
    }

    /// Scroll up
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    /// Scroll down
    pub fn scroll_down(&mut self) {
        self.scroll_offset += 1;
    }

    /// Clear messages
    pub fn clear(&mut self) {
        self.messages.clear();
        self.scroll_offset = 0;
    }
}

impl Default for AiDialogue {
    fn default() -> Self {
        Self::new()
    }
}
