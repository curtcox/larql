//! Patch statement parsers: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH.

use super::{ParseError, Parser};
use crate::ast::*;
use crate::lexer::{Keyword, Token};

impl Parser {
    /// Parse a statement starting with BEGIN (BEGIN PATCH "file.vlp").
    pub(crate) fn parse_begin(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Begin)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::BeginPatch { path })
    }

    /// Parse SAVE PATCH.
    pub(crate) fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Save)?;
        self.expect_keyword(Keyword::Patch)?;
        self.eat_semicolon();
        Ok(Statement::SavePatch)
    }

    /// Parse APPLY PATCH "file.vlp".
    pub(crate) fn parse_apply(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Apply)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::ApplyPatch { path })
    }

    /// Parse REMOVE PATCH "file.vlp".
    pub(crate) fn parse_remove(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Remove)?;
        self.expect_keyword(Keyword::Patch)?;
        let path = self.expect_string()?;
        self.eat_semicolon();
        Ok(Statement::RemovePatch { path })
    }

    /// Parse ATTACH CALL FROM FILE "call_patch.json".
    pub(crate) fn parse_attach(&mut self) -> Result<Statement, ParseError> {
        self.expect_keyword(Keyword::Attach)?;
        self.expect_keyword(Keyword::Call)?;
        if self.check_keyword(Keyword::From) {
            self.advance();
            self.expect_ident_eq("file")?;
            let path = self.expect_string()?;
            self.eat_semicolon();
            return Ok(Statement::AttachCall { path });
        }

        self.expect_keyword(Keyword::At)?;
        self.expect_keyword(Keyword::Layer)?;
        let layer = self.expect_u32()? as usize;
        self.expect_ident_eq("feature")?;
        let feature = self.expect_u32()? as usize;

        self.expect_ident_eq("gate")?;
        self.expect_ident_eq("vector")?;
        self.expect_keyword(Keyword::From)?;
        self.expect_ident_eq("file")?;
        let gate_vector_path = self.expect_string()?;

        self.expect_ident_eq("monty")?;
        self.expect_ident_eq("code")?;
        self.expect_keyword(Keyword::From)?;
        self.expect_ident_eq("file")?;
        let monty_code_path = self.expect_string()?;

        let mut score_threshold = None;
        let mut margin_threshold = None;
        let mut max_calls_per_token = None;
        let mut time_us = None;
        let mut memory_bytes = None;
        let mut steps = None;

        loop {
            match self.peek() {
                Token::Ident(ref s) if s.eq_ignore_ascii_case("trigger") => {
                    self.advance();
                    loop {
                        match self.peek() {
                            Token::Ident(ref s) if s.eq_ignore_ascii_case("score") => {
                                self.advance();
                                if matches!(self.peek(), Token::Gte | Token::Eq) {
                                    self.advance();
                                }
                                score_threshold = Some(self.expect_f32()?);
                            }
                            Token::Ident(ref s) if s.eq_ignore_ascii_case("margin") => {
                                self.advance();
                                if matches!(self.peek(), Token::Gte | Token::Eq) {
                                    self.advance();
                                }
                                margin_threshold = Some(self.expect_f32()?);
                            }
                            Token::Ident(ref s)
                                if s.eq_ignore_ascii_case("max_calls_per_token") =>
                            {
                                self.advance();
                                max_calls_per_token = Some(self.expect_u32()? as usize);
                            }
                            _ => break,
                        }
                    }
                }
                Token::Ident(ref s) if s.eq_ignore_ascii_case("limits") => {
                    self.advance();
                    loop {
                        match self.peek() {
                            Token::Ident(ref s) if s.eq_ignore_ascii_case("time_us") => {
                                self.advance();
                                time_us = Some(self.expect_u32()? as u64);
                            }
                            Token::Ident(ref s) if s.eq_ignore_ascii_case("memory_bytes") => {
                                self.advance();
                                memory_bytes = Some(self.expect_u32()? as u64);
                            }
                            Token::Ident(ref s) if s.eq_ignore_ascii_case("steps") => {
                                self.advance();
                                steps = Some(self.expect_u32()? as u64);
                            }
                            _ => break,
                        }
                    }
                }
                _ => break,
            }
        }

        self.eat_semicolon();
        Ok(Statement::AttachCallInline(AttachCallInline {
            layer,
            feature,
            gate_vector_path,
            monty_code_path,
            score_threshold,
            margin_threshold,
            max_calls_per_token,
            time_us,
            memory_bytes,
            steps,
        }))
    }
}
